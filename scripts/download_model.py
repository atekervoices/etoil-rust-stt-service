#!/usr/bin/env python3
"""
Download Luganda fine-tuned phosai/nemotron-3.5-asr-streaming-luganda model (.nemo) from Hugging Face.
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


def download_nemotron_model(target_dir="./models", token=None):
    os.makedirs(target_dir, exist_ok=True)
    repo_id = os.environ.get("MODEL_REPO_ID", "phosai/nemotron-3.5-asr-streaming-luganda")
    filename = os.environ.get("MODEL_FILENAME", "nemotron-asr-streaming-luganda.nemo")
    hf_token = token or os.environ.get("HF_TOKEN") or None

    print(f"Downloading {filename} from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
    )
    print(f"✅ Nemotron Luganda model downloaded successfully to: {file_path}")
    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download phosai/nemotron-3.5-asr-streaming-luganda model"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save the downloaded .nemo file (default: ./models)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token for private repository",
    )
    args = parser.parse_args()
    download_nemotron_model(args.output_dir, args.token)

