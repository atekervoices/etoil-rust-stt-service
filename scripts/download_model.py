#!/usr/bin/env python3
"""
Download official nvidia/nemotron-3.5-asr-streaming-0.6b model (.nemo) from Hugging Face.
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


def download_nemotron_model(target_dir="./models"):
    os.makedirs(target_dir, exist_ok=True)
    repo_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    filename = "nemotron-3.5-asr-streaming-0.6b.nemo"

    print(f"Downloading {filename} from {repo_id}...")
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print(f"✅ NVIDIA Nemotron model downloaded successfully to: {file_path}")
    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download official NVIDIA Nemotron 3.5 Streaming model"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save the downloaded .nemo file (default: ./models)",
    )
    args = parser.parse_args()
    download_nemotron_model(args.output_dir)
