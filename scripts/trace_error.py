import os
import traceback
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT

hf_token = os.environ.get("HF_TOKEN")
model_id = "phosai/nemotron-3.5-asr-streaming-luganda"

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

enc = EncoderWrapper(model).eval()
feats = inputs["input_features"]
mask = inputs["attention_mask"]
prompts = inputs["prompt_ids"]

try:
    with torch.inference_mode():
        torch.jit.trace(enc, (feats, mask, prompts))
except Exception:
    traceback.print_exc()
