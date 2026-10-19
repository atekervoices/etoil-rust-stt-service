import os
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForRNNT

hf_token = os.environ.get("HF_TOKEN")
model_id = "phosai/nemotron-3.5-asr-streaming-luganda"

print("Loading...")
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token).eval()

audio_path = "/export/audio/audio_1.16k.wav"
audio, sr = sf.read(audio_path)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="lug")
print("Input keys:", inputs.keys())
print("input_features shape:", inputs["input_features"].shape)
if "attention_mask" in inputs:
    print("attention_mask shape:", inputs["attention_mask"].shape)
if "prompt_ids" in inputs:
    print("prompt_ids:", inputs["prompt_ids"])

with torch.inference_mode():
    outputs = model.generate(**inputs)
    print("outputs.sequences:", outputs.sequences)
    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print("Transcription:", text)
