import os
import json
import torch
import torch.nn as nn
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT

raw_token = os.environ.get("HF_TOKEN", "").strip()
hf_token = raw_token if raw_token else None
model_id = os.environ.get("MODEL_ID", "").strip()
if not model_id:
    raise ValueError(
        "MODEL_ID environment variable is required. Pass it via docker-compose.yml (MODEL_ID) or --build-arg MODEL_ID=..."
    )
lookahead_tokens = int(os.environ.get("LOOKAHEAD_TOKENS", "3").strip() or "3")
output_dir = os.environ.get("OUTPUT_DIR", "/export/torchscript_model")
os.makedirs(output_dir, exist_ok=True)

if hf_token is None:
    print("Notice: HF_TOKEN is not set. Accessing Hugging Face without auth token...")
else:
    print(f"Using provided HF_TOKEN (length: {len(hf_token)})...")

print(f"Loading model & processor from {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token).eval()
fe = processor.feature_extractor

# 1. Save tokenizer, model config & prompt dictionary
processor.save_pretrained(output_dir)
model.config.save_pretrained(output_dir)

prompt_dict = getattr(processor, "prompt_dictionary", {"lug": 63, "en": 0, "en-US": 0})
with open(os.path.join(output_dir, "prompt_dictionary.json"), "w") as f:
    json.dump(prompt_dict, f, indent=2)

blank_id = int(getattr(model.config, "blank_token_id", getattr(model.config, "vocab_size", 2048)))
dec_hidden = int(getattr(model.decoder.lstm, "hidden_size", 640))
dec_layers = int(getattr(model.decoder.lstm, "num_layers", 2))
print(f"Saved tokenizer, config.json, and prompt_dictionary.json (blank_id={blank_id}, dec_layers={dec_layers}, dec_hidden={dec_hidden})")

LOG_ZERO_GUARD_VALUE = 5.960464477539063e-08

# 2. End-to-End Encoder Wrapper taking raw 16kHz audio waveform [1, N]
class EndToEndEncoderWrapper(nn.Module):
    def __init__(self, m, fe, num_lookahead_tokens: int = 3):
        super().__init__()
        self.encoder = m.encoder
        self.prompt_projector = m.prompt_projector
        self.encoder_projector = m.encoder_projector
        self.num_prompts = m.config.num_prompts

        # Default num_lookahead_tokens = 3 (320ms chunk attention for Pass 1; Pass 2 dynamically overrides chunk_size to 14 = 1120ms)
        if hasattr(m.encoder, "_resolve_attn_context"):
            left_ctx, right_ctx = m.encoder._resolve_attn_context(num_lookahead_tokens)
        else:
            left_ctx, right_ctx = 56, num_lookahead_tokens
        self.left_ctx_frames = max(left_ctx, 56)
        self.chunk_size = right_ctx + 1
        self.left_context_chunks = self.left_ctx_frames // self.chunk_size
        lookahead_ms = self.chunk_size * 80
        print(
            f"Configured dynamic multi-pass lookahead: Pass 1 default={lookahead_ms}ms (chunk_size={self.chunk_size}), "
            f"Pass 2 endpoint rescoring=1120ms (chunk_size=14), left_ctx_frames={self.left_ctx_frames}"
        )

        # Mel extraction parameters
        self.win_length = fe.win_length
        self.hop_length = fe.hop_length
        self.n_fft = fe.n_fft
        self.register_buffer("mel_filters", fe.mel_filters)
        self.register_buffer("window", torch.hann_window(fe.win_length, periodic=False))
        self.log_zero_guard = LOG_ZERO_GUARD_VALUE

    def extract_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        # Preemphasis: x[0], x[t] - 0.97*x[t-1]
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

        # Zero out padding frames matching HF feature extractor
        audio_length = waveform.shape[1]
        feature_length = audio_length // self.hop_length
        T_frames = input_features.shape[1]
        frame_idx = torch.cumsum(input_features.new_ones(1, T_frames, dtype=torch.long), dim=-1) - 1
        mask = frame_idx < feature_length
        input_features = input_features * mask.unsqueeze(-1)
        return input_features

    def forward(self, waveform: torch.Tensor, prompt_ids: torch.Tensor, chunk_size: torch.Tensor):
        # 1. Feature extraction from raw audio waveform [1, N]
        input_features = self.extract_fbank(waveform)

        # 2. Subsampling
        embeds = self.encoder.subsampling(input_features, None)
        embeds = embeds * self.encoder.input_scale
        T = embeds.shape[1]

        # 3. Dynamic 2D chunked causal mask supporting both Pass 1 (chunk_size=4, 320ms) and Pass 2 (chunk_size=14, 1120ms)
        cs = torch.clamp(chunk_size[0], min=1)
        left_chunks = torch.clamp(torch.div(self.left_ctx_frames, cs, rounding_mode="trunc"), min=1)
        idx = torch.cumsum(embeds.new_ones(T, dtype=torch.long), dim=0) - 1
        q = idx[:, None]
        k = idx[None, :]
        q_chunk = torch.div(q, cs, rounding_mode="trunc")
        kv_chunk = torch.div(k, cs, rounding_mode="trunc")
        diff = q_chunk - kv_chunk
        mask = (diff >= 0) & (diff <= left_chunks)
        attention_mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        all_masked_rows = torch.all(~attention_mask, dim=-1)

        # 4. Position embeddings (device-agnostic relative sinusoidal positional encoding)
        hidden_size = embeds.shape[-1]
        seq_length = embeds.shape[1]
        num_pos = 2 * seq_length - 1
        ramp_pos = torch.cumsum(embeds.new_ones(num_pos, dtype=torch.long), dim=0) - 1
        pos_ids = ((seq_length - 1) - ramp_pos).float()[None, None, :]
        ramp_dim = (torch.cumsum(embeds.new_ones(hidden_size // 2), dim=0) - 1).float()
        inv_f = (1.0 / (10000.0 ** (2.0 * ramp_dim / hidden_size)))[None, :, None].expand(embeds.shape[0], -1, -1)
        freqs = (inv_f @ pos_ids).transpose(1, 2)
        pos_emb = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(embeds.shape[0], num_pos, -1).to(dtype=embeds.dtype)

        # 5. Conformer Layers
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

# 3. Decoder Wrapper (uses LSTMCell per layer to avoid cuDNN _cudnn_rnn weight compaction warnings on GPU)
class DecoderWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.embedding = m.decoder.embedding
        self.decoder_projector = m.decoder.decoder_projector
        src_lstm = m.decoder.lstm
        self.num_layers = src_lstm.num_layers
        cells = []
        for i in range(self.num_layers):
            input_size = src_lstm.input_size if i == 0 else src_lstm.hidden_size
            cell = nn.LSTMCell(input_size, src_lstm.hidden_size, bias=src_lstm.bias)
            cell.weight_ih.data.copy_(getattr(src_lstm, f"weight_ih_l{i}").data)
            cell.weight_hh.data.copy_(getattr(src_lstm, f"weight_hh_l{i}").data)
            if src_lstm.bias:
                cell.bias_ih.data.copy_(getattr(src_lstm, f"bias_ih_l{i}").data)
                cell.bias_hh.data.copy_(getattr(src_lstm, f"bias_hh_l{i}").data)
            cells.append(cell)
        self.cells = nn.ModuleList(cells)

    def forward(self, input_ids: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        x = self.embedding(input_ids).squeeze(1)
        new_h_list = []
        new_c_list = []
        for i, cell in enumerate(self.cells):
            hx, cx = cell(x, (h[i], c[i]))
            x = hx
            new_h_list.append(hx)
            new_c_list.append(cx)
        new_h = torch.stack(new_h_list, dim=0)
        new_c = torch.stack(new_c_list, dim=0)
        dec_out = self.decoder_projector(x.unsqueeze(1))
        return dec_out, new_h, new_c

# 4. Joint Wrapper
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

# Trace and Export
audio_path = "/export/audio/audio_1.16k.wav"
if os.path.exists(audio_path):
    print(f"Reading sample audio from {audio_path} for tracing...")
    audio, sr = sf.read(audio_path)
    raw_waveform = torch.from_numpy(audio).float().unsqueeze(0)
else:
    print("Using synthetic 16kHz audio waveform (1.0s) for tracing...")
    raw_waveform = torch.randn(1, 16000, dtype=torch.float32)

prompts = torch.tensor([63], dtype=torch.long)
default_chunk_tensor = torch.tensor([lookahead_tokens + 1], dtype=torch.long)

with torch.inference_mode():
    print("Tracing EndToEndEncoderWrapper...")
    e2e_enc = EndToEndEncoderWrapper(model, fe, num_lookahead_tokens=lookahead_tokens).eval()
    traced_enc = torch.jit.trace(e2e_enc, (raw_waveform, prompts, default_chunk_tensor))
    enc_path = os.path.join(output_dir, "encoder.pt")
    traced_enc.save(enc_path)
    print(f"Saved {enc_path}")

    print("Tracing DecoderWrapper...")
    dec_wrap = DecoderWrapper(model).eval()
    dummy_input_ids = torch.tensor([[blank_id]], dtype=torch.long)
    dummy_h = torch.zeros(dec_layers, 1, dec_hidden)
    dummy_c = torch.zeros(dec_layers, 1, dec_hidden)
    traced_dec = torch.jit.trace(dec_wrap, (dummy_input_ids, dummy_h, dummy_c))
    dec_path = os.path.join(output_dir, "decoder.pt")
    traced_dec.save(dec_path)
    print(f"Saved {dec_path}")

    print("Tracing JointWrapper...")
    joint_wrap = JointWrapper(model).eval()
    dummy_dec_out = torch.randn(1, 1, dec_hidden)
    dummy_frame = torch.randn(1, 1, dec_hidden)
    traced_joint = torch.jit.trace(joint_wrap, (dummy_dec_out, dummy_frame))
    joint_path = os.path.join(output_dir, "joint.pt")
    traced_joint.save(joint_path)
    print(f"Saved {joint_path}")

print("Export completed successfully!")
