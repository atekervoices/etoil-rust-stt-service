import soundfile as sf
import torch
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

traced_enc = torch.jit.load("/export/output_torchscript/encoder.pt")
traced_dec = torch.jit.load("/export/output_torchscript/decoder.pt")
traced_joint = torch.jit.load("/export/output_torchscript/joint.pt")

audio, sr = sf.read("/export/audio/audio_1.16k.wav")
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("/export/output_torchscript")
inputs = p(audio, sampling_rate=16000, return_tensors="pt", language="lug")

audio_feats = traced_enc(inputs["input_features"], inputs["attention_mask"], inputs["prompt_ids"])
num_frames = audio_feats.shape[1]
print("Total acoustic frames:", num_frames)

blank_id = 13087
h = torch.zeros(2, 1, 640)
c = torch.zeros(2, 1, 640)
last_token = torch.tensor([[blank_id]], dtype=torch.long)
dec_out, h, c = traced_dec(last_token, h, c)

frame_idx = 0
symbols_at_frame = 0
emitted = []

while frame_idx < num_frames and len(emitted) < 500:
    frame = audio_feats[:, frame_idx:frame_idx+1, :]
    logits = traced_joint(dec_out, frame)
    token = int(torch.argmax(logits, dim=-1).item())
    
    if token == blank_id or symbols_at_frame >= 10:
        frame_idx += 1
        symbols_at_frame = 0
    else:
        emitted.append(token)
        symbols_at_frame += 1
        last_token = torch.tensor([[token]], dtype=torch.long)
        dec_out, h, c = traced_dec(last_token, h, c)

transcript = p.tokenizer.decode(emitted, skip_special_tokens=True)
print("\n" + "="*60)
print("EXACT MATCH TRANSCRIPTION:")
print(f"'{transcript}'")
print("="*60)
