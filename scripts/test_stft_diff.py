import numpy as np
import torch

# Create a test signal
np.random.seed(42)
x_np = np.random.randn(16000).astype(np.float32)
x_torch = torch.from_numpy(x_np)

n_fft = 512
hop_length = 160
win_length = 400
window = torch.hann_window(win_length)

# 1. Torch STFT (ground truth matching NeMo)
stft_torch = torch.stft(x_torch, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, return_complex=True)
mag_torch = (torch.abs(stft_torch) ** 2).numpy()

# 2. Rust audio.rs logic
pad_amount = n_fft // 2
padded = np.pad(x_np, (pad_amount, pad_amount), mode='constant')
window_np = window.numpy()
num_frames = len(x_np) // hop_length
window_offset = (n_fft - win_length) // 2

spec_rust_as_written = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)
spec_rust_fixed = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)

for f in range(num_frames):
    start = f * hop_length
    
    # As written in audio.rs:
    inp_1 = np.zeros(n_fft, dtype=np.float32)
    for i in range(win_length):
        inp_1[window_offset + i] = padded[start + window_offset + i] * window_np[i]
    fft_1 = np.fft.rfft(inp_1)
    spec_rust_as_written[:, f] = np.abs(fft_1) ** 2

    # Fixed:
    inp_2 = np.zeros(n_fft, dtype=np.float32)
    for i in range(win_length):
        inp_2[window_offset + i] = padded[start + i] * window_np[i]
    fft_2 = np.fft.rfft(inp_2)
    spec_rust_fixed[:, f] = np.abs(fft_2) ** 2

diff_written = np.max(np.abs(mag_torch[:, :num_frames] - spec_rust_as_written))
diff_fixed = np.max(np.abs(mag_torch[:, :num_frames] - spec_rust_fixed))

print(f"Max diff with code as written in audio.rs: {diff_written:.4e}")
print(f"Max diff with fixed (padded[start + i]):  {diff_fixed:.4e}")
