import torch
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import chunked_limited_mask_function
from transformers import AutoModelForRNNT
import os

hf_token = os.environ.get("HF_TOKEN")
model_id = "phosai/nemotron-3.5-asr-streaming-luganda"
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token).eval()

T = 93
num_lookahead_tokens = 3
left_ctx, right_ctx = model.encoder._resolve_attn_context(num_lookahead_tokens)
print(f"left_ctx: {left_ctx}, right_ctx: {right_ctx}")
chunk_size = right_ctx + 1
left_context_chunks = left_ctx // chunk_size if left_ctx >= 0 else 10_000
print(f"chunk_size: {chunk_size}, left_context_chunks: {left_context_chunks}")

# Vectorized mask
q = torch.arange(T)[:, None]
k = torch.arange(T)[None, :]
q_chunk = torch.div(q, chunk_size, rounding_mode="trunc")
kv_chunk = torch.div(k, chunk_size, rounding_mode="trunc")
diff = q_chunk - kv_chunk
vec_mask = (diff >= 0) & (diff <= left_context_chunks) # shape [T, T]
vec_mask = vec_mask[None, None, :, :] # shape [1, 1, T, T]

# Model's mask
from transformers.masking_utils import create_bidirectional_mask
dummy_embeds = torch.randn(1, T, 640)
model_mask = create_bidirectional_mask(
    config=model.config,
    inputs_embeds=dummy_embeds,
    attention_mask=None,
    and_mask_function=chunked_limited_mask_function(left_ctx, right_ctx),
)

print(f"vec_mask shape: {vec_mask.shape}, model_mask shape: {model_mask.shape}")
print(f"Exact match: {torch.equal(vec_mask, model_mask)}")
if not torch.equal(vec_mask, model_mask):
    print("Diff count:", (vec_mask != model_mask).sum().item())
