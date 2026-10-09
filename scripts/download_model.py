#!/usr/bin/env python3
"""
Download CraneAILabs/crane-nemo-asr-1.1 model from Hugging Face.
"""

import os
import sys
import argparse

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it: pip install huggingface_hub")
    sys.exit(1)


def download_crane_model(target_dir="./models"):
    os.makedirs(target_dir, exist_ok=True)
    repo_id = "CraneAILabs/crane-nemo-asr-1.1"
    filename = "crane-nemo-asr-1.1.nemo"

    print(f"Downloading {filename} from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Model downloaded successfully to: {file_path}")
    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Crane Nemotron 1.1 model")
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save the downloaded .nemo file (default: ./models)"
    )
    args = parser.parse_args()
    download_crane_model(args.output_dir)
