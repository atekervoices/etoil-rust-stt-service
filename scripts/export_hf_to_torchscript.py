import os
import sys
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT
import transformers.masking_utils as mu

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Patch sdpa_mask to avoid vmap inside TorchScript tracing
orig_sdpa_mask = mu.sdpa_mask
def patched_sdpa_mask(*args, **kwargs):
    kwargs["use_vmap"] = False
    return orig_sdpa_mask(*args, **kwargs)
mu.sdpa_mask = patched_sdpa_mask

def main():
    model_id = "phosai/nemotron-3.5-asr-streaming-luganda"
    hf_token = os.environ.get("HF_TOKEN")
    output_dir = os.environ.get("OUTPUT_DIR", "/export/output_torchscript")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model & processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model = AutoModelForRNNT.from_pretrained(
        model_id,
        dtype=torch.float32,
        attn_implementation="eager",
        token=hf_token
    ).eval()
    print("Model loaded successfully.")

    # Save tokenizer and processor config
    processor.save_pretrained(output_dir)

    # 1. Encoder Wrapper
    class EncoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.encoder
            self.prompt_projector = m.prompt_projector
            self.encoder_projector = m.encoder_projector
            self.num_prompts = m.config.num_prompts

        def forward(self, input_features, attention_mask, prompt_ids):
            enc_out = self.encoder(input_features=input_features, attention_mask=attention_mask)
            h = enc_out.last_hidden_state
            one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.num_prompts).to(h.dtype)
            one_hot = one_hot[:, None, :].expand(-1, h.shape[1], -1)
            fused = self.prompt_projector(torch.cat([h, one_hot], dim=-1))
            return self.encoder_projector(fused)

    # 2. Decoder Wrapper
    class DecoderWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.decoder = m.decoder

        def forward(self, input_ids, h, c):
            emb = self.decoder.embedding(input_ids)
            lstm_out, (new_h, new_c) = self.decoder.lstm(emb, (h, c))
            dec_out = self.decoder.decoder_projector(lstm_out)
            return dec_out, new_h, new_c

    # 3. Joint Wrapper
    class JointWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.joint = m.joint

        def forward(self, decoder_out, audio_frame):
            logits = self.joint(
                decoder_hidden_states=decoder_out[:, None, :, :],
                encoder_hidden_states=audio_frame[:, :, None, :]
            )
            return logits.squeeze(1).squeeze(1)

    print("Tracing Encoder...")
    dummy_feats = torch.randn(1, 93, 128)
    dummy_mask = torch.ones(1, 93, dtype=torch.long)
    dummy_prompt = torch.tensor([63], dtype=torch.long)
    with torch.inference_mode():
        traced_enc = torch.jit.trace(EncoderWrapper(model).eval(), (dummy_feats, dummy_mask, dummy_prompt))
        enc_path = os.path.join(output_dir, "encoder.pt")
        traced_enc.save(enc_path)
        print(f"Saved {enc_path}")

        print("Tracing Decoder...")
        dummy_input_ids = torch.tensor([[13087]], dtype=torch.long)
        dummy_h = torch.zeros(2, 1, 640)
        dummy_c = torch.zeros(2, 1, 640)
        traced_dec = torch.jit.trace(DecoderWrapper(model).eval(), (dummy_input_ids, dummy_h, dummy_c))
        dec_path = os.path.join(output_dir, "decoder.pt")
        traced_dec.save(dec_path)
        print(f"Saved {dec_path}")

        print("Tracing Joint...")
        dummy_dec_out = torch.randn(1, 1, 640)
        dummy_audio_frame = torch.randn(1, 1, 640)
        traced_joint = torch.jit.trace(JointWrapper(model).eval(), (dummy_dec_out, dummy_audio_frame))
        joint_path = os.path.join(output_dir, "joint.pt")
        traced_joint.save(joint_path)
        print(f"Saved {joint_path}")

        # Verification on audio_1.16k.wav
        audio_path = "/export/audio/audio_1.16k.wav"
        if not os.path.exists(audio_path):
            audio_path = "/export/audio/audio (1).wav"

        print(f"\nVerifying TorchScript models against {audio_path}...")
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        import torchaudio.functional as F
        waveform = torch.from_numpy(audio).float()
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz...")
            waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)
        audio_16k = waveform.numpy()

        inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt", language="lug")
        audio_feats = traced_enc(inputs["input_features"], inputs["attention_mask"], inputs["prompt_ids"])
        print(f"Acoustic features frames: {audio_feats.shape[1]}")

        blank_id = 13087
        h = torch.zeros(2, 1, 640)
        c = torch.zeros(2, 1, 640)
        last_token = torch.tensor([[blank_id]], dtype=torch.long)
        dec_out, h, c = traced_dec(last_token, h, c)

        emitted = []
        num_frames = audio_feats.shape[1]
        for t in range(num_frames):
            frame = audio_feats[:, t:t+1, :]
            for _ in range(10):
                logits = traced_joint(dec_out, frame)
                token = int(torch.argmax(logits, dim=-1).item())
                if token == blank_id:
                    break
                emitted.append(token)
                last_token = torch.tensor([[token]], dtype=torch.long)
                dec_out, h, c = traced_dec(last_token, h, c)

        transcript = processor.tokenizer.decode(emitted, skip_special_tokens=True)
        print("\n" + "="*60)
        print("TORCHSCRIPT VERIFICATION TRANSCRIPT:")
        print(f"'{transcript}'")
        print("="*60)

if __name__ == "__main__":
    main()
