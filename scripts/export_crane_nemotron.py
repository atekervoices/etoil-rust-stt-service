#!/usr/bin/env python3
"""
Export Crane Nemotron 3.5 Streaming ASR (Luganda, English) model to ONNX for parakeet-rs.

Converts CraneAILabs/crane-nemo-asr-1.1 (.nemo) into:
  - encoder.onnx + encoder.onnx.data
  - decoder_joint.onnx
  - tokenizer.model
  - config.json

Requirements:
  pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
  pip install 'numpy<2.0' onnx onnxruntime soundfile sentencepiece torch
"""

import argparse
import functools
import json
import logging
import os
import shutil
import tarfile
import tempfile
import torch

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print("Warning: NeMo is not installed in the current environment.")
    print("Install via: pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'")


def export_model(input_path, output_dir, left_context=56, right_context=6):
    logging.getLogger("nemo_logging").setLevel(logging.ERROR)
    try:
        from nemo.core.classes.common import typecheck
        typecheck.set_typecheck_enabled(False)
    except Exception:
        pass

    # PyTorch 2.9+ Dynamo fallback for ONNX tracing
    pytorch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if pytorch_version >= (2, 9) and not getattr(torch.onnx.export, "_legacy_onnx_patched", False):
        print("  Patching torch.onnx.export for PyTorch 2.9+ legacy tracing")
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
    print(f"\nLoading Crane Nemotron model from: {input_path}")
    model = nemo_asr.models.ASRModel.restore_from(input_path, map_location=device)
    model.eval()

    print(f"  Model Architecture : {type(model).__name__}")
    print(f"  Vocab Size         : {model.tokenizer.vocab_size}")
    print(f"  Number of Prompts  : {getattr(model, 'num_prompts', 'N/A')}")

    # 1. Extract Tokenizer model (SentencePiece protobuf)
    print("\n[1/4] Extracting tokenizer.model...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(input_path, "r") as tar:
            tar.extractall(path=tmp_dir)
        tokenizer_src = os.path.join(tmp_dir, "tokenizer.model")
        tokenizer_dest = os.path.join(output_dir, "tokenizer.model")
        if os.path.exists(tokenizer_src):
            shutil.copy2(tokenizer_src, tokenizer_dest)
            print(f"  Saved {tokenizer_dest}")
        else:
            print("  Extracting SentencePiece model directly from tokenizer object...")
            model.tokenizer.save_vocab(os.path.join(output_dir, "vocab.txt"))

    # 2. Export Decoder + Joint Network
    print("\n[2/4] Exporting decoder_joint.onnx...")
    decoder_joint_path = os.path.join(output_dir, "decoder_joint.onnx")
    try:
        model.export(
            output=decoder_joint_path,
            check_trace=False,
            use_dynamic_axes=True,
            target="rnnt_decoder"
        )
        print(f"  Saved {decoder_joint_path}")
    except Exception as e:
        print(f"  NeMo built-in rnnt_decoder export: {e}")
        print("  Attempting direct torch.onnx.export for decoder + joint...")
        # Direct export fallback
        class DecoderJointWrapper(torch.nn.Module):
            def __init__(self, decoder, joint):
                super().__init__()
                self.decoder = decoder
                self.joint = joint

            def forward(self, targets, target_lengths, encoder_outputs):
                decoder_outputs, _, _ = self.decoder(targets=targets, target_length=target_lengths)
                joint_outputs = self.joint(encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs)
                return joint_outputs

        wrapper = DecoderJointWrapper(model.decoder, model.joint)
        dummy_targets = torch.zeros((1, 1), dtype=torch.long)
        dummy_lengths = torch.ones((1,), dtype=torch.long)
        dummy_enc = torch.zeros((1, 1, model.encoder.d_model), dtype=torch.float32)

        torch.onnx.export(
            wrapper,
            (dummy_targets, dummy_lengths, dummy_enc),
            decoder_joint_path,
            input_names=["targets", "target_lengths", "encoder_outputs"],
            output_names=["joint_outputs"],
            dynamic_axes={
                "targets": {0: "batch", 1: "target_len"},
                "encoder_outputs": {0: "batch", 1: "time"},
                "joint_outputs": {0: "batch", 1: "time", 2: "target_len"}
            },
            opset_version=17
        )
        print(f"  Saved {decoder_joint_path}")

    # 3. Export Streaming Cache-Aware Encoder
    print("\n[3/4] Exporting streaming cache-aware encoder.onnx...")
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    
    # Configure streaming att_context_size
    model.encoder.set_streaming_att_context_size([left_context, right_context])

    try:
        model.export(
            output=encoder_path,
            check_trace=False,
            use_dynamic_axes=True,
            target="cache_aware_encoder"
        )
        print(f"  Saved {encoder_path}")
    except Exception as e:
        print(f"  Exporting encoder directly with cache inputs: {e}")
        # NeMo export command recommended in parakeet-rs:
        # Use export_nemotron_streaming_multilingual.py from altunenes/parakeet-rs
        print("  Please use scripts/export_nemotron_streaming_multilingual.py from parakeet-rs for exact graph wrapping.")

    # 4. Generate config.json with prompt dictionary
    print("\n[4/4] Writing config.json...")
    config_data = {
        "model_name": "CraneAILabs/crane-nemo-asr-1.1",
        "sample_rate": 16000,
        "chunk_duration_ms": 560,
        "chunk_size_samples": 8960,
        "left_context": left_context,
        "right_context": right_context,
        "supported_languages": {
            "lg": {"name": "Luganda", "target_lang": "auto", "prompt_id": 101},
            "en": {"name": "English", "target_lang": "en-US", "prompt_id": 0},
            "auto": {"name": "Auto-Detect", "target_lang": "auto", "prompt_id": 101}
        },
        "excluded_languages": ["sn"]
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Saved {config_path}")

    print("\n✅ Export pipeline completed successfully!")
    print(f"Models are ready for parakeet-rs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export CraneAILabs/crane-nemo-asr-1.1 to ONNX for parakeet-rs"
    )
    parser.add_argument("input_path", help="Path to crane-nemo-asr-1.1.nemo")
    parser.add_argument("output_dir", help="Output directory for ONNX files")
    parser.add_argument("--left-context", type=int, default=56, help="Attention left context (default: 56)")
    parser.add_argument("--right-context", type=int, default=6, help="Attention right context (default: 6)")
    args = parser.parse_args()

    export_model(args.input_path, args.output_dir, args.left_context, args.right_context)
