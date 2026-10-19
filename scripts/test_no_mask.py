import os
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

    def forward(self, input_features, prompt_ids):
        # Pass attention_mask=None
        enc_out = self.encoder(
            input_features=input_features,
            attention_mask=None,
            num_lookahead_tokens=3
        )
        h = enc_out.last_hidden_state
        one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.num_prompts).to(h.dtype)
        one_hot = one_hot[:, None, :].expand(-1, h.shape[1], -1)
        fused = self.prompt_projector(torch.cat([h, one_hot], dim=-1))
        return self.encoder_projector(fused)

enc = EncoderWrapper(model).eval()
feats = inputs["input_features"]
prompts = inputs["prompt_ids"]

print("1. Running un-traced EncoderWrapper with attention_mask=None...")
with torch.inference_mode():
    expected_feats = enc(feats, prompts)

print("2. Attempting torch.jit.trace with attention_mask=None...")
try:
    with torch.inference_mode():
        traced_enc = torch.jit.trace(enc, (feats, prompts))
        print("Tracing SUCCEEDED!")
        out_traced = traced_enc(feats, prompts)
        diff = (expected_feats - out_traced).abs().max().item()
        print(f"Max diff: {diff}")
        
        # Now let's test if we decode with this audio_feats!
        blank_id = model.config.blank_token_id
        num_layers = model.decoder.lstm.num_layers
        hidden_size = model.decoder.lstm.hidden_size
        h_lstm = torch.zeros(num_layers, 1, hidden_size)
        c_lstm = torch.zeros(num_layers, 1, hidden_size)
        last_token = torch.tensor([[blank_id]], dtype=torch.long)
        emb = model.decoder.embedding(last_token)
        lstm_out, (h_lstm, c_lstm) = model.decoder.lstm(emb, (h_lstm, c_lstm))
        dec_out = model.decoder.decoder_projector(lstm_out)

        emitted = []
        T = out_traced.shape[1]
        for t in range(T):
            frame = out_traced[:, t:t+1, :]
            for _ in range(10):
                logits = model.joint(
                    decoder_hidden_states=dec_out[:, None, :, :],
                    encoder_hidden_states=frame[:, :, None, :]
                ).squeeze(1).squeeze(1)
                token = int(torch.argmax(logits, dim=-1).item())
                if token == blank_id:
                    break
                emitted.append(token)
                last_token = torch.tensor([[token]], dtype=torch.long)
                emb = model.decoder.embedding(last_token)
                lstm_out, (h_lstm, c_lstm) = model.decoder.lstm(emb, (h_lstm, c_lstm))
                dec_out = model.decoder.decoder_projector(lstm_out)

        transcript = processor.tokenizer.decode(emitted, skip_special_tokens=True)
        print("\n" + "="*60)
        print(f"TRANSCRIPT FROM TRACED ENCODER: '{transcript}'")
        print("="*60)
except Exception as e:
    import traceback
    traceback.print_exc()
