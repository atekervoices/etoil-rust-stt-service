import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT
import transformers.masking_utils as mu

# Patch sdpa_mask to disable functorch vmap inside TorchScript tracing
orig_sdpa_mask = mu.sdpa_mask
def patched_sdpa_mask(*args, **kwargs):
    kwargs["use_vmap"] = False
    return orig_sdpa_mask(*args, **kwargs)
mu.sdpa_mask = patched_sdpa_mask

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    model_id = "phosai/nemotron-3.5-asr-streaming-luganda"
    hf_token = os.environ.get("HF_TOKEN")
    output_dir = os.environ.get("OUTPUT_DIR", "/export/output_onnx")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Hugging Face model and processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    print("Processor loaded successfully. Loading AutoModelForRNNT weights...")
    model = AutoModelForRNNT.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager", token=hf_token).eval()
    print("Model loaded successfully into memory.")

    # Save tokenizer and configs for Rust loader
    processor.save_pretrained(output_dir)
    print(f"Saved processor/tokenizer configs to {output_dir}")

    # =========================================================================
    # Wrapper 1: Acoustic Encoder + Prompt Projector
    # =========================================================================
    class HFEncoderExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.encoder
            self.prompt_projector = model.prompt_projector
            self.encoder_projector = model.encoder_projector
            self.num_prompts = model.config.num_prompts

        def forward(self, input_features, attention_mask, prompt_ids):
            # input_features: [B, T, 128], attention_mask: [B, T], prompt_ids: [B]
            encoder_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)
            hidden_states = encoder_outputs.last_hidden_state
            
            # Broadcast one-hot prompt vector
            one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.num_prompts).to(hidden_states.dtype)
            one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
            fused = self.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
            
            # Projected acoustic representations [B, T_sub, 640]
            return self.encoder_projector(fused)

    encoder_wrapper = HFEncoderExport(model).eval()

    B, T = 1, 800
    dummy_features = torch.randn(B, T, 128)
    dummy_mask = torch.ones(B, T, dtype=torch.long)
    dummy_prompt = torch.tensor([63], dtype=torch.long)

    encoder_onnx_path = os.path.join(output_dir, "encoder.onnx")
    print(f"Exporting encoder to {encoder_onnx_path}...")
    torch.onnx.export(
        encoder_wrapper,
        (dummy_features, dummy_mask, dummy_prompt),
        encoder_onnx_path,
        input_names=["input_features", "attention_mask", "prompt_ids"],
        output_names=["audio_features"],
        dynamic_axes={
            "input_features": {1: "time"},
            "attention_mask": {1: "time"},
            "audio_features": {1: "subsampled_time"}
        },
        opset_version=17,
        dynamo=False,
    )
    print("Encoder export complete.")

    # =========================================================================
    # Wrapper 2: RNNT Decoder + Joint Network
    # =========================================================================
    class HFDecoderJointExport(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.decoder
            self.joint = model.joint

        def forward(self, audio_frame, input_ids, hidden_state, cell_state):
            # audio_frame: [B, 1, 640], input_ids: [B, 1], hidden/cell: [2, B, 640]
            embeddings = self.decoder.embedding(input_ids)
            lstm_out, (new_h, new_c) = self.decoder.lstm(embeddings, (hidden_state, cell_state))
            decoder_out = self.decoder.decoder_projector(lstm_out)
            
            logits = self.joint(decoder_out, audio_frame)
            return logits.squeeze(1), new_h, new_c

    decoder_wrapper = HFDecoderJointExport(model).eval()

    dummy_audio_frame = torch.randn(1, 1, 640)
    dummy_input_id = torch.tensor([[13087]], dtype=torch.long)
    dummy_h = torch.zeros(2, 1, 640)
    dummy_c = torch.zeros(2, 1, 640)

    decoder_onnx_path = os.path.join(output_dir, "decoder_joint.onnx")
    print(f"Exporting decoder_joint to {decoder_onnx_path}...")
    torch.onnx.export(
        decoder_wrapper,
        (dummy_audio_frame, dummy_input_id, dummy_h, dummy_c),
        decoder_onnx_path,
        input_names=["audio_frame", "input_ids", "hidden_state", "cell_state"],
        output_names=["logits", "new_hidden_state", "new_cell_state"],
        opset_version=17,
        dynamo=False,
    )
    print("Decoder-joint export complete.")

    # =========================================================================
    # Verification with ONNX Runtime
    # =========================================================================
    audio_path = "/export/audio/audio (1).wav"
    if not os.path.exists(audio_path):
        audio_path = "audio/audio (1).wav"

    if os.path.exists(audio_path):
        print(f"\nVerifying ONNX export against {audio_path}...")
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        import torchaudio.functional as F
        waveform = torch.from_numpy(audio).float()
        if sr != 16000:
            print(f"Resampling audio from {sr}Hz to 16000Hz...")
            waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)
        audio_16k = waveform.numpy()

        inputs = processor(audio_16k, sampling_rate=16000, return_tensors="np", language="lug")

        enc_session = ort.InferenceSession(encoder_onnx_path, providers=["CPUExecutionProvider"])
        dec_session = ort.InferenceSession(decoder_onnx_path, providers=["CPUExecutionProvider"])

        in_feats = np.array(inputs["input_features"], dtype=np.float32)
        in_mask = np.array(inputs["attention_mask"], dtype=np.int64)
        raw_prompt = inputs.get("prompt_ids", 63)
        if isinstance(raw_prompt, np.ndarray):
            in_prompt = raw_prompt.astype(np.int64).flatten()
            if len(in_prompt) == 0 or in_prompt.ndim == 0:
                in_prompt = np.array([63], dtype=np.int64)
        elif isinstance(raw_prompt, (list, tuple)):
            in_prompt = np.array(raw_prompt, dtype=np.int64)
        else:
            in_prompt = np.array([int(raw_prompt)], dtype=np.int64)

        enc_outs = enc_session.run(None, {
            "input_features": in_feats,
            "attention_mask": in_mask,
            "prompt_ids": in_prompt
        })
        audio_features = enc_outs[0]

        blank_id = 13087
        h = np.zeros((2, 1, 640), dtype=np.float32)
        c = np.zeros((2, 1, 640), dtype=np.float32)
        last_token = blank_id
        emitted_tokens = []

        num_frames = audio_features.shape[1]
        print(f"Acoustic frames to decode: {num_frames}")

        for t in range(num_frames):
            frame = audio_features[:, t:t+1, :]
            for _ in range(10):
                dec_outs = dec_session.run(None, {
                    "audio_frame": frame,
                    "input_ids": np.array([[last_token]], dtype=np.int64),
                    "hidden_state": h,
                    "cell_state": c,
                })
                logits, new_h, new_c = dec_outs[0], dec_outs[1], dec_outs[2]
                token_id = int(np.argmax(logits[0]))
                if token_id == blank_id:
                    break
                emitted_tokens.append(token_id)
                last_token = token_id
                h, c = new_h, new_c

        transcript = processor.tokenizer.decode(emitted_tokens, skip_special_tokens=True)
        print("\n" + "="*60)
        print("ONNX VERIFICATION TRANSCRIPT:")
        print(f"'{transcript}'")
        print("="*60)

if __name__ == "__main__":
    main()
