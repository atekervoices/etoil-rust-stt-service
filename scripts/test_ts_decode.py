import torch
import librosa
from transformers import AutoProcessor
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("Loading TorchScript models from torchscript_model/...", flush=True)
enc = torch.jit.load("torchscript_model/encoder.pt", map_location="cpu").eval()
dec = torch.jit.load("torchscript_model/decoder.pt", map_location="cpu").eval()
joint = torch.jit.load("torchscript_model/joint.pt", map_location="cpu").eval()
processor = AutoProcessor.from_pretrained("torchscript_model")

def decode(audio_path, prompt_id, lang_name):
    audio, sr = librosa.load(audio_path, sr=16000)
    wav_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    prompt_tensor = torch.tensor([prompt_id], dtype=torch.long)
    
    with torch.no_grad():
        audio_feats = enc(wav_tensor, prompt_tensor)
        num_frames = audio_feats.shape[1]
        
        blank_id = 13087
        h = torch.zeros(2, 1, 640)
        c = torch.zeros(2, 1, 640)
        last_token = torch.tensor([[blank_id]], dtype=torch.long)
        dec_out, h, c = dec(last_token, h, c)
        
        frame_idx = 0
        symbols_at_frame = 0
        emitted = []
        
        # Also collect logits statistics
        argmax_counts = {}
        
        while frame_idx < num_frames and len(emitted) < 500:
            frame = audio_feats[:, frame_idx:frame_idx+1, :]
            logits = joint(dec_out, frame)
            token = int(torch.argmax(logits, dim=-1).item())
            
            argmax_counts[token] = argmax_counts.get(token, 0) + 1
            
            if token == blank_id or symbols_at_frame >= 10:
                frame_idx += 1
                symbols_at_frame = 0
            else:
                emitted.append(token)
                symbols_at_frame += 1
                last_token = torch.tensor([[token]], dtype=torch.long)
                dec_out, h, c = dec(last_token, h, c)
                
        text = processor.tokenizer.decode(emitted, skip_special_tokens=True)
        top_tokens = sorted(argmax_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"[{audio_path}] Prompt {prompt_id} ({lang_name}): emitted={len(emitted)}, frames={num_frames}, top_tokens={top_tokens}")
        print(f"--> Result: '{text}'\n", flush=True)

for p_id, name in [(63, "lug"), (0, "en (0)"), (1, "en-GB (1)"), (101, "auto (101)")]:
    decode("../test.wav", p_id, name)
    decode("english_test.wav", p_id, name)
