import os
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT

hf_token = os.environ.get("HF_TOKEN")
model_id = "phosai/nemotron-3.5-asr-streaming-luganda"

print("Loading model...")
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token).eval()

audio_path = "/export/audio/audio_1.16k.wav"
audio, sr = sf.read(audio_path)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="lug")

class EncoderWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
        self.prompt_projector = m.prompt_projector
        self.encoder_projector = m.encoder_projector
        self.num_prompts = m.config.num_prompts

    def forward(self, input_features, attention_mask, prompt_ids):
        enc_out = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=3
        )
        h = enc_out.last_hidden_state
        one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.num_prompts).to(h.dtype)
        one_hot = one_hot[:, None, :].expand(-1, h.shape[1], -1)
        fused = self.prompt_projector(torch.cat([h, one_hot], dim=-1))
        return self.encoder_projector(fused)

class DecoderWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.decoder = m.decoder

    def forward(self, input_ids, h, c):
        emb = self.decoder.embedding(input_ids)
        lstm_out, (new_h, new_c) = self.decoder.lstm(emb, (h, c))
        dec_out = self.decoder.decoder_projector(lstm_out)
        return dec_out, new_h, new_c

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

enc_wrap = EncoderWrapper(model).eval()
dec_wrap = DecoderWrapper(model).eval()
joint_wrap = JointWrapper(model).eval()

feats = inputs["input_features"]
mask = inputs["attention_mask"]
prompts = inputs["prompt_ids"]

print(f"Feats shape: {feats.shape}, mask shape: {mask.shape}, prompts shape: {prompts.shape}")

print("Testing un-traced EncoderWrapper output...")
with torch.inference_mode():
    expected_feats = enc_wrap(feats, mask, prompts)
    print("expected_feats shape:", expected_feats.shape)

print("Attempting to trace EncoderWrapper...")
try:
    with torch.inference_mode():
        # Trace with the real input shapes!
        traced_enc = torch.jit.trace(enc_wrap, (feats, mask, prompts))
        print("Tracing succeeded!")
        out_traced = traced_enc(feats, mask, prompts)
        diff = (expected_feats - out_traced).abs().max().item()
        print(f"Max diff between original and traced encoder: {diff}")
except Exception as e:
    print(f"Tracing failed with error: {e}")
