import os
import torch
import librosa
from transformers import AutoProcessor, AutoModelForRNNT

model_id = "phosai/nemotron-3.5-asr-streaming-luganda"
hf_token = os.environ.get("HF_TOKEN")

print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = AutoModelForRNNT.from_pretrained(model_id, token=hf_token)
model.eval()

for wav_path, test_lang in [("english_test.wav", "en"), ("../test.wav", "lug"), ("english_test.wav", "lug")]:
    audio, sr = librosa.load(wav_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=test_lang)
    print(f"\nTesting {wav_path} with language='{test_lang}':")
    print(f"prompt_ids: {inputs.get('prompt_ids')}")
    
    with torch.inference_mode():
        outputs = model.generate(**inputs)
    
    transcription = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    print(f"Result: '{transcription}'")
