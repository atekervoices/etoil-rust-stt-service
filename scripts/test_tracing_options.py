import os
import inspect
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT

hf_token = os.environ.get("HF_TOKEN")
model_id = "phosai/nemotron-3.5-asr-streaming-luganda"

processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token).eval()

audio_path = "/export/audio/audio_1.16k.wav"
audio, sr = sf.read(audio_path)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="lug")

print("model class:", type(model))
print("model.encoder.forward args:", inspect.signature(model.encoder.forward))

# Let's see how model.forward or model.generate does encoder forward
enc_out = model.encoder(
    input_features=inputs["input_features"],
    attention_mask=inputs.get("attention_mask"),
    num_lookahead_tokens=inputs.get("num_lookahead_tokens", 3)
)
h = enc_out.last_hidden_state
prompt_ids = inputs["prompt_ids"]
num_prompts = model.config.num_prompts
one_hot = torch.nn.functional.one_hot(prompt_ids, num_classes=num_prompts).to(h.dtype)
one_hot = one_hot[:, None, :].expand(-1, h.shape[1], -1)
fused = model.prompt_projector(torch.cat([h, one_hot], dim=-1))
audio_feats = model.encoder_projector(fused)

print("audio_feats shape:", audio_feats.shape)

blank_id = model.config.blank_token_id
num_layers = model.decoder.lstm.num_layers
hidden_size = model.decoder.lstm.hidden_size
print(f"blank_id: {blank_id}, num_layers: {num_layers}, hidden_size: {hidden_size}")

h_lstm = torch.zeros(num_layers, 1, hidden_size)
c_lstm = torch.zeros(num_layers, 1, hidden_size)
last_token = torch.tensor([[blank_id]], dtype=torch.long)

emb = model.decoder.embedding(last_token)
lstm_out, (h_lstm, c_lstm) = model.decoder.lstm(emb, (h_lstm, c_lstm))
dec_out = model.decoder.decoder_projector(lstm_out)

emitted = []
T = audio_feats.shape[1]

for t in range(T):
    frame = audio_feats[:, t:t+1, :]
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
print("MANUAL GREEDY TRANSCRIPT WITH num_lookahead_tokens=3:")
print(f"'{transcript}'")
print("="*60)
