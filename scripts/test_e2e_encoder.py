import os
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT

hf_token = os.environ.get("HF_TOKEN")
model_id = "phosai/nemotron-3.5-asr-streaming-luganda"
output_dir = os.environ.get("OUTPUT_DIR", "/export/torchscript_model")
os.makedirs(output_dir, exist_ok=True)

print("Loading HuggingFace model...")
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token).eval()
fe = processor.feature_extractor

LOG_ZERO_GUARD_VALUE = 5.960464477539063e-08

class EndToEndEncoderWrapper(nn.Module):
    def __init__(self, m, fe):
        super().__init__()
        self.encoder = m.encoder
        self.prompt_projector = m.prompt_projector
        self.encoder_projector = m.encoder_projector
        self.num_prompts = m.config.num_prompts
        self.chunk_size = 4
        self.left_context_chunks = 14

        # Mel extraction parameters
        self.win_length = fe.win_length
        self.hop_length = fe.hop_length
        self.n_fft = fe.n_fft
        self.register_buffer("mel_filters", fe.mel_filters)
        self.register_buffer("window", torch.hann_window(fe.win_length, periodic=False))
        self.log_zero_guard = LOG_ZERO_GUARD_VALUE

    def extract_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        # Preemphasis: first sample untouched, remainder: x[t] - 0.97*x[t-1]
        preemp = torch.cat([waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1)

        stft = torch.stft(
            preemp,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
            center=True,
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        mel_spec = self.mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + self.log_zero_guard)
        input_features = mel_spec.permute(0, 2, 1) # [B, T, D]

        # Zero out padding frames exactly as Hugging Face feature extractor does
        audio_length = waveform.shape[1]
        feature_length = audio_length // self.hop_length
        T_frames = input_features.shape[1]
        mask = torch.arange(T_frames, device=waveform.device)[None, :] < feature_length
        input_features = input_features * mask.unsqueeze(-1)
        return input_features

    def forward(self, waveform: torch.Tensor, prompt_ids: torch.Tensor):
        # 1. Feature extraction from raw audio waveform [1, N]
        input_features = self.extract_fbank(waveform)

        # 2. Subsampling
        embeds = self.encoder.subsampling(input_features, None)
        embeds = embeds * self.encoder.input_scale
        T = embeds.shape[1]

        # 3. Dynamic 2D chunked causal mask without vmap
        device = embeds.device
        q = torch.arange(T, device=device)[:, None]
        k = torch.arange(T, device=device)[None, :]
        q_chunk = torch.div(q, self.chunk_size, rounding_mode="trunc")
        kv_chunk = torch.div(k, self.chunk_size, rounding_mode="trunc")
        diff = q_chunk - kv_chunk
        mask = (diff >= 0) & (diff <= self.left_context_chunks)
        attention_mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        all_masked_rows = torch.all(~attention_mask, dim=-1)

        # 4. Position embeddings
        pos_emb = self.encoder.encode_positions(embeds, cached_frames=0)

        # 5. Layers
        hidden_states = embeds
        for layer in self.encoder.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                all_masked_rows=all_masked_rows,
                position_embeddings=pos_emb,
            )

        # 6. Prompt projection & Encoder projection
        one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.num_prompts).to(hidden_states.dtype)
        one_hot = one_hot[:, None, :].expand(-1, T, -1)
        fused = self.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
        return self.encoder_projector(fused)

audio_path = "/export/audio/audio_1.16k.wav"
audio, sr = sf.read(audio_path)
raw_waveform = torch.from_numpy(audio).float().unsqueeze(0) # [1, num_samples]
prompts = torch.tensor([63], dtype=torch.long)

e2e_enc = EndToEndEncoderWrapper(model, fe).eval()

print("Testing EndToEndEncoderWrapper on raw waveform...")
with torch.inference_mode():
    feats = e2e_enc(raw_waveform, prompts)
    print(f"feats shape: {feats.shape}")

    print("Tracing EndToEndEncoderWrapper...")
    traced_e2e_enc = torch.jit.trace(e2e_enc, (raw_waveform, prompts))
    enc_path = os.path.join(output_dir, "encoder.pt")
    traced_e2e_enc.save(enc_path)
    print(f"Saved {enc_path}")

print("Testing variable audio sample lengths on exported encoder.pt...")
reloaded_enc = torch.jit.load(enc_path)
for num_samples in [16000, 32000, 48000, len(audio)]:
    dummy_wav = torch.randn(1, num_samples)
    dummy_out = reloaded_enc(dummy_wav, prompts)
    print(f"{num_samples} samples ({num_samples/16000:.1f}s) -> encoded shape: {dummy_out.shape}")

# Test full greedy decode with decoder.pt & joint.pt
dec_wrap = torch.jit.load(os.path.join(output_dir, "decoder.pt"))
joint_wrap = torch.jit.load(os.path.join(output_dir, "joint.pt"))

blank_id = 13087
h_lstm = torch.zeros(2, 1, 640)
c_lstm = torch.zeros(2, 1, 640)
last_token = torch.tensor([[blank_id]], dtype=torch.long)
dec_out, h_lstm, c_lstm = dec_wrap(last_token, h_lstm, c_lstm)

emitted = []
enc_out = reloaded_enc(raw_waveform, prompts)
T = enc_out.shape[1]

for t in range(T):
    frame = enc_out[:, t:t+1, :]
    for _ in range(10):
        logits = joint_wrap(dec_out, frame)
        token = int(torch.argmax(logits, dim=-1).item())
        if token == blank_id:
            break
        emitted.append(token)
        last_token = torch.tensor([[token]], dtype=torch.long)
        dec_out, h_lstm, c_lstm = dec_wrap(last_token, h_lstm, c_lstm)

transcript = processor.tokenizer.decode(emitted, skip_special_tokens=True)
print("\n" + "="*70)
print(f"END-TO-END WAVEFORM-IN TRANSCRIPT:\n'{transcript}'")
print("="*70)
