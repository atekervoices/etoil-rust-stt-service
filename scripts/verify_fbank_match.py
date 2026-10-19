import torch
import soundfile as sf
from transformers import AutoProcessor

hf_token = os.environ.get("HF_TOKEN", "")
processor = AutoProcessor.from_pretrained("phosai/nemotron-3.5-asr-streaming-luganda", token=hf_token)
fe = processor.feature_extractor

audio, sr = sf.read("/export/audio/audio_1.16k.wav")
expected = fe(audio, sampling_rate=16000, return_tensors="pt")["input_features"]

# Manual implementation matching fe.__call__
raw_w = torch.from_numpy(audio).float().unsqueeze(0)
preemp_w = torch.cat([raw_w[:, :1], raw_w[:, 1:] - 0.97 * raw_w[:, :-1]], dim=1)
manual = fe._torch_extract_fbank_features(preemp_w, "cpu", center=True)

diff = (expected - manual).abs().max().item()
print(f"Max diff between manual fbank and official feature extractor: {diff}")
