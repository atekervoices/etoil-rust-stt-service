import os
import sys
import json
import soundfile as sf
import torch

try:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

def run_tests():
    model_dir = "/export/models"
    os.makedirs(model_dir, exist_ok=True)
    nemo_path = os.path.join(model_dir, "nemotron-asr-streaming-luganda.nemo")

    if not os.path.exists(nemo_path):
        print(f"Downloading model from phosai/nemotron-3.5-asr-streaming-luganda...")
        hf_token = os.environ.get("HF_TOKEN")
        nemo_path = hf_hub_download(
            repo_id="phosai/nemotron-3.5-asr-streaming-luganda",
            filename="nemotron-asr-streaming-luganda.nemo",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

    print(f"\n==================================================")
    print(f"Loading .nemo model from: {nemo_path}")
    print(f"==================================================")

    # Load model on CPU
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    model.eval()

    print(f"\nModel class: {model.__class__.__name__}")
    
    # Check config
    if hasattr(model, "cfg"):
        print("\n--- Model Defaults Config ---")
        if "model_defaults" in model.cfg:
            print("model_defaults:")
            for k, v in model.cfg.model_defaults.items():
                if k == "prompt_dictionary":
                    print(f"  prompt_dictionary: {len(v)} entries")
                    # Check for luganda or similar
                    lug_entries = {k2: v2 for k2, v2 in v.items() if any(sub in k2.lower() for sub in ["lug", "ganda", "lg", "sw", "en", "auto"])}
                    print(f"  Sample prompt entries: {lug_entries}")
                else:
                    print(f"  {k}: {v}")
        
        if "decoding" in model.cfg:
            print("\ndecoding config:")
            print(OmegaConf.to_yaml(model.cfg.decoding))

    audio_file = "/export/audio/audio (1).wav"
    if not os.path.exists(audio_file):
        audio_file = "audio/audio (1).wav"
    
    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file {audio_file} not found!")
        return

    sig, sr = sf.read(audio_file)
    print(f"\nInput audio: {audio_file} (duration: {len(sig)/sr:.2f}s, sr: {sr})")

    # Test 1: Native transcribe (offline/batch mode)
    print("\n--- Test 1: Native model.transcribe() [default] ---")
    try:
        res = model.transcribe([audio_file])
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error in Test 1: {e}")

    # Test 2: If model has set_inference_prompt
    if hasattr(model, "set_inference_prompt"):
        test_prompts = ["auto", "en-US", "sw-KE"]
        # Check if 'lug' or 'lug-UG' is in prompt_dictionary
        p_dict = model.cfg.model_defaults.get("prompt_dictionary", {})
        if "lug" in p_dict:
            test_prompts.insert(0, "lug")
        if "lug-UG" in p_dict:
            test_prompts.insert(0, "lug-UG")

        for p in test_prompts:
            print(f"\n--- Test with inference prompt: '{p}' ---")
            try:
                model.set_inference_prompt(p)
                res = model.transcribe([audio_file])
                print(f"Result ({p}): {res}")
            except Exception as e:
                print(f"Error with prompt {p}: {e}")

    # Test 3: Check decoding strategy
    print("\n==================================================")
    print("Testing complete.")

if __name__ == "__main__":
    run_tests()
