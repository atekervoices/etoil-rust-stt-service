#!/usr/bin/env python3
"""
Export NVIDIA Nemotron 3.5 Streaming ASR (multilingual, 0.6B) to ONNX for parakeet-rs.
Specifically tailored for Crane AI Labs' crane-nemo-asr-1.1 model (Luganda, English).

Exports:
  - encoder.onnx + encoder.onnx.data (streaming FastConformer + prompt-kernel head)
  - decoder_joint.onnx (RNNT decoder + joint network)
  - tokenizer.model (SentencePiece tokenizer, vocab ~13k)
  - config.json (streaming params, cache shapes, prompt_dictionary)
"""

import argparse
import functools
import gc
import glob
import json
import logging
import os
import shutil
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Nemotron 3.5 streaming ASR model (multilingual) to ONNX for parakeet-rs"
    )
    parser.add_argument("input_path", help="Path to .nemo model file")
    parser.add_argument("output_dir", help="Directory for ONNX outputs")
    parser.add_argument(
        "--left-context",
        type=int,
        default=56,
        help="Attention left context (default: 56)",
    )
    parser.add_argument(
        "--right-context",
        type=int,
        default=6,
        help="Attention right context (default: 6; valid values: 0, 3, 6, 13)",
    )
    parser.add_argument(
        "--verify-langs",
        default="en-US,auto",
        help="Comma-separated language codes to verify ONNX output matches NeMo",
    )
    return parser.parse_args()


class EncoderWithPromptWrapper(torch.nn.Module):
    """
    Wraps the streaming encoder and inlines NeMo's prompt kernel,
    exposing prompt_index as an ONNX input tensor for multilingual inference in parakeet-rs.
    """

    def __init__(self, model, drop_extra):
        super().__init__()
        self.encoder = model.encoder
        self.prompt_kernel = model.prompt_kernel
        self.num_prompts = model.num_prompts
        self.drop_extra = drop_extra

    def forward(
        self,
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        prompt_index,
    ):
        encoded, enc_len, ch_n, tm_n, ln_n = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )

        # Mirror _apply_prompt_to_encoded:
        # encoder output is (B, D, T), transpose to (B, T, D),
        # build [B, T, num_prompts] one-hot, concat, run prompt_kernel, transpose back.
        encoded = encoded.transpose(1, 2)
        B, T, _ = encoded.shape
        prompt = torch.zeros(
            B, T, self.num_prompts, dtype=encoded.dtype, device=encoded.device
        )
        prompt.scatter_(
            2,
            prompt_index.view(B, 1, 1).expand(-1, T, -1),
            1.0,
        )
        encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
        encoded = encoded.transpose(1, 2)
        return encoded, enc_len, ch_n, tm_n, ln_n


def extract_tokenizer(input_path, output_dir, model=None):
    """Robustly extracts tokenizer.model from .nemo tar archive or model object."""
    tokenizer_dest = os.path.join(output_dir, "tokenizer.model")

    # Method 1: Search members of .nemo tarball
    try:
        with tarfile.open(input_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("tokenizer.model"):
                    f = tar.extractfile(member)
                    if f is not None:
                        with open(tokenizer_dest, "wb") as out:
                            out.write(f.read())
                        print(f"  Extracted tokenizer.model from archive member: {member.name}")
                        return True
    except Exception as e:
        print(f"  Tar extraction attempt note: {e}")

    # Method 2: Inspect model tokenizer attributes if loaded
    if model is not None and hasattr(model, "tokenizer"):
        for attr in ["model_path", "tokenizer_file", "model"]:
            path_val = getattr(model.tokenizer, attr, None)
            if isinstance(path_val, str) and os.path.exists(path_val):
                shutil.copy2(path_val, tokenizer_dest)
                print(f"  Copied tokenizer.model from tokenizer.{attr}: {path_val}")
                return True

    return os.path.exists(tokenizer_dest)


def main():
    args = parse_args()
    att_context_size = [args.left_context, args.right_context]
    output_dir = args.output_dir
    verify_langs = [s.strip() for s in args.verify_langs.split(",") if s.strip()]

    # Mute noisy NeMo logging
    logging.getLogger("nemo_logging").setLevel(logging.ERROR)
    try:
        from nemo.core.classes.common import typecheck
        typecheck.set_typecheck_enabled(False)
    except Exception:
        pass

    # PyTorch 2.9+ dynamo export patch
    pytorch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    print(f"PyTorch version: {torch.__version__}")
    _patch_marker = "_legacy_onnx_patched"
    if pytorch_version >= (2, 9) and not getattr(torch.onnx.export, _patch_marker, False):
        print("  Patching torch.onnx.export for PyTorch 2.9+ (dynamo=False)")
        _original_export = torch.onnx.export

        @functools.wraps(_original_export)
        def _patched_export(*pargs, **kwargs):
            if "dynamo" not in kwargs:
                kwargs["dynamo"] = False
            return _original_export(*pargs, **kwargs)

        _patched_export._legacy_onnx_patched = True
        torch.onnx.export = _patched_export

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"\nLoading model from: {args.input_path}")
    model = nemo_asr.models.ASRModel.restore_from(args.input_path, map_location=device)
    model.eval()

    print(f"  Model architecture: {type(model).__name__}")
    print(f"  Vocab size        : {model.tokenizer.vocab_size}")
    print(f"  Num prompts       : {getattr(model, 'num_prompts', 'N/A')}")

    # Step 1: Extract tokenizer
    print("\n[1/4] Extracting tokenizer.model...")
    if not extract_tokenizer(args.input_path, output_dir, model):
        raise RuntimeError(f"Failed to extract tokenizer.model from {args.input_path}")
    print(f"  Saved {os.path.join(output_dir, 'tokenizer.model')}")

    # Step 2: Configure streaming parameters
    print("\nConfiguring streaming parameters...")
    if hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(att_context_size)

    streaming_cfg = model.encoder.streaming_cfg
    subsampling_factor = model.cfg.encoder.get("subsampling_factor", 8)
    left_context = att_context_size[0]
    right_context = att_context_size[1]
    chunk_size = right_context + 1
    model.encoder.setup_streaming_params(chunk_size=chunk_size, shift_size=chunk_size)
    drop_extra_pre_encoded = getattr(streaming_cfg, "drop_extra_pre_encoded", 0)

    print(f"  Subsampling factor     : {subsampling_factor}")
    print(f"  Left context           : {left_context}")
    print(f"  Right context          : {right_context}")
    print(f"  Chunk size (frames)    : {chunk_size}")
    print(f"  drop_extra_pre_encoded : {drop_extra_pre_encoded}")

    # Initial cache state
    batch_size = 1
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=batch_size)
    )
    num_layers = cache_last_channel.shape[0]
    hidden_dim = cache_last_channel.shape[3]
    conv_context = cache_last_time.shape[3]

    # Sample mel features via streaming buffer
    print("\nGenerating sample mel features...")
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=True,
    )
    sample_rate = 16000
    dummy_audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, dummy_audio, sample_rate)
    temp_wav.close()

    streaming_buffer.append_audio_file(temp_wav.name, stream_id=-1)
    processed_signal, processed_signal_length = next(iter(streaming_buffer))
    os.unlink(temp_wav.name)

    # Prompt dictionary
    prompt_dict = OmegaConf.to_container(
        model.cfg.model_defaults.prompt_dictionary, resolve=True
    )
    print(f"  prompt_dictionary: {len(prompt_dict)} entries")

    # Step 3: Export Encoder with prompt_index input
    print("\n[2/4] Exporting streaming cache-aware encoder.onnx...")
    default_lang = "en-US" if "en-US" in prompt_dict else list(prompt_dict.keys())[0]
    model.set_inference_prompt(default_lang)
    prompt_idx_t = torch.tensor([model._inference_prompt_index], dtype=torch.long)

    with torch.no_grad():
        enc_raw, encoded_len, _, _, _ = model.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )
        enc_ref = model._apply_prompt_to_encoded(enc_raw)

    wrapper = EncoderWithPromptWrapper(model, drop_extra_pre_encoded).eval()
    with torch.no_grad():
        enc_wrap, _, _, _, _ = wrapper(
            processed_signal,
            processed_signal_length,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            prompt_idx_t,
        )
    diff = (enc_wrap - enc_ref).abs().max().item()
    print(f"  Wrapper vs NeMo max diff: {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError(f"Wrapper output deviates from NeMo: diff={diff:.2e}")

    temp_encoder_path = os.path.join(output_dir, "encoder_temp.onnx")
    input_names = [
        "processed_signal",
        "processed_signal_length",
        "cache_last_channel",
        "cache_last_time",
        "cache_last_channel_len",
        "prompt_index",
    ]
    output_names = [
        "encoded",
        "encoded_len",
        "cache_last_channel_next",
        "cache_last_time_next",
        "cache_last_channel_len_next",
    ]

    torch.onnx.export(
        wrapper,
        (
            processed_signal,
            processed_signal_length,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            prompt_idx_t,
        ),
        temp_encoder_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes={
            "processed_signal": {0: "batch", 2: "time"},
            "processed_signal_length": {0: "batch"},
            "prompt_index": {0: "batch"},
            "encoded": {0: "batch", 2: "time"},
            "encoded_len": {0: "batch"},
        },
    )

    # Consolidate encoder weights into external data file
    print("  Consolidating encoder weights into encoder.onnx + encoder.onnx.data...")
    encoder_model = onnx.load(temp_encoder_path, load_external_data=True)
    final_encoder_path = os.path.join(output_dir, "encoder.onnx")

    pre_encode_cache = int(getattr(streaming_cfg, "pre_encode_cache_size", [0, 9])[1])
    encoder_model.metadata_props.append(
        onnx.StringStringEntryProto(key="chunk_size_output_frames", value=str(chunk_size))
    )
    encoder_model.metadata_props.append(
        onnx.StringStringEntryProto(key="pre_encode_cache", value=str(pre_encode_cache))
    )

    onnx.save_model(
        encoder_model,
        final_encoder_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.onnx.data",
        size_threshold=0,
    )
    del encoder_model
    gc.collect()

    # Clean up temporary encoder fragments
    for pattern in ["encoder_temp*", "*.weight", "*MatMul*", "Constant_*", "onnx__*", "encoder.pre_encode*"]:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"  Saved {final_encoder_path} and encoder.onnx.data")

    # Step 4: Export Decoder / Joint via NeMo's native exporter
    print("\n[3/4] Exporting decoder_joint.onnx via NeMo exporter...")
    temp_decoder_prefix = os.path.join(output_dir, "temp_model")
    with torch.no_grad():
        model.export(output=temp_decoder_prefix + ".onnx", check_trace=False)

    final_decoder_path = os.path.join(output_dir, "decoder_joint.onnx")
    found_decoder = False

    for f in glob.glob(os.path.join(output_dir, "*.onnx")):
        fname = os.path.basename(f).lower()
        if "decoder" in fname and "joint" in fname:
            if f != final_decoder_path:
                shutil.move(f, final_decoder_path)
            found_decoder = True
            break

    if not found_decoder:
        for f in glob.glob(os.path.join(output_dir, "*.onnx")):
            fname = os.path.basename(f).lower()
            if ("decoder" in fname or "joint" in fname) and "encoder" not in fname:
                if f != final_decoder_path:
                    shutil.move(f, final_decoder_path)
                found_decoder = True
                break

    if not found_decoder:
        remaining = [os.path.basename(x) for x in glob.glob(os.path.join(output_dir, "*"))]
        raise RuntimeError(f"Could not find exported decoder/joint ONNX in {output_dir}. Files found: {remaining}")

    # Remove temporary files not in our final set
    keep = {"encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model"}
    for f in glob.glob(os.path.join(output_dir, "*")):
        if os.path.basename(f) not in keep and os.path.isfile(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"  Saved {final_decoder_path}")

    # Step 5: Save config.json
    print("\n[4/4] Generating config.json...")
    config = {
        "model_name": "crane-nemo-asr-1.1",
        "sample_rate": 16000,
        "n_mels": 128,
        "subsampling_factor": subsampling_factor,
        "att_context_size": att_context_size,
        "left_context": left_context,
        "right_context": right_context,
        "chunk_size_output_frames": chunk_size,
        "pre_encode_cache": pre_encode_cache,
        "drop_extra_pre_encoded": drop_extra_pre_encoded,
        "num_encoder_layers": num_layers,
        "hidden_dim": hidden_dim,
        "conv_context": conv_context,
        "vocab_size": model.tokenizer.vocab_size,
        "blank_id": model.tokenizer.vocab_size,
        "num_prompts": int(model.num_prompts),
        "prompt_dictionary": prompt_dict,
        "preprocessor": {
            "window_size": 0.025,
            "window_stride": 0.01,
            "n_fft": 512,
            "normalize": "NA",
            "preemph": 0.97,
        },
        "cache_shapes": {
            "cache_last_channel": list(cache_last_channel.shape),
            "cache_last_time": list(cache_last_time.shape),
            "cache_last_channel_len": [1],
        },
        "test_input": {
            "mel_shape": list(processed_signal.shape),
            "mel_length": int(processed_signal_length[0]),
            "prompt_index": int(prompt_idx_t[0]),
        },
        "test_output": {
            "encoded_shape": list(enc_ref.shape),
            "encoded_len": int(encoded_len[0]),
        },
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {config_path}")

    # Step 6: Quick ONNX verification
    print("\nVerifying exported ONNX graphs...")
    enc_session = ort.InferenceSession(final_encoder_path, providers=["CPUExecutionProvider"])
    for lang in verify_langs:
        if lang not in prompt_dict:
            print(f"  [skip] '{lang}' not in prompt_dictionary")
            continue
        idx = prompt_dict[lang]
        model.set_inference_prompt(lang)
        with torch.no_grad():
            e_raw, _, _, _, _ = model.encoder.cache_aware_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            e_ref = model._apply_prompt_to_encoded(e_raw).numpy()

        e_onnx = enc_session.run(
            None,
            {
                "processed_signal": processed_signal.numpy(),
                "processed_signal_length": processed_signal_length.numpy(),
                "cache_last_channel": cache_last_channel.numpy(),
                "cache_last_time": cache_last_time.numpy(),
                "cache_last_channel_len": cache_last_channel_len.numpy(),
                "prompt_index": np.array([idx], dtype=np.int64),
            },
        )[0]
        d = np.abs(e_ref - e_onnx).max()
        print(f"  Verified {lang} (prompt {idx}): max ONNX diff = {d:.2e}")

    del enc_session
    dec_session = ort.InferenceSession(final_decoder_path, providers=["CPUExecutionProvider"])
    print(f"  Verified decoder_joint.onnx session initialized successfully.")
    del dec_session
    gc.collect()

    print("\n" + "=" * 60)
    print("SUCCESS: Crane Nemotron ONNX export complete!")
    print("=" * 60)
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024**2)
            print(f"  {fname:25s} ({size_mb:6.1f} MB)")


if __name__ == "__main__":
    main()
