import torch

traced_enc = torch.jit.load("/export/output_torchscript/encoder.pt")
print("Testing variable length input on traced_enc...")
for L in [300, 500, 732, 1000]:
    dummy = torch.randn(1, L, 128)
    prompt = torch.tensor([63], dtype=torch.long)
    try:
        out = traced_enc(dummy, prompt)
        print(f"Input len {L} -> Output shape: {out.shape}")
    except Exception as e:
        print(f"Input len {L} FAILED: {e}")
