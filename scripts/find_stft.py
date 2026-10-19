import numpy as np
import torch

# Let's inspect exact PyTorch STFT behavior
x = torch.randn(1, 16000)
n_fft = 512
hop_length = 160
win_length = 400
window = torch.hann_window(win_length)

# Torch STFT
stft_torch = torch.stft(x[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, pad_mode='reflect', return_complex=True)
print(f"Torch STFT shape: {stft_torch.shape}")

# Let's see how PyTorch pads:
# PyTorch center=True does:
pad_amount = n_fft // 2
x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad_amount, pad_amount), mode='reflect').squeeze(1)

# And then unfolds:
# In torch.stft, frames are extracted at step hop_length, length win_length or n_fft?
# Let's check:
for mode in ['reflect', 'constant']:
    for offset in range(0, 100):
        # test matching first frame
        frame_torch = stft_torch[:, 0]
        # try to reconstruct frame 0
        sl = x_padded[0, offset:offset+win_length] * window
        padded_sl = torch.zeros(n_fft)
        # place in center
        wo = (n_fft - win_length) // 2
        padded_sl[wo:wo+win_length] = sl
        fft_calc = torch.fft.rfft(padded_sl)
        diff = torch.max(torch.abs(frame_torch - fft_calc)).item()
        if diff < 1e-4:
            print(f"Exact match found! mode={mode}, offset={offset}, window_offset={wo}, diff={diff}")
            break
