#!/usr/bin/env python3
"""
Download PHOSAI Nemotron 3.5 Streaming ASR Luganda model (.nemo) from Hugging Face.
Supports private repositories using HF_TOKEN / HUGGING_FACE_HUB_TOKEN or --token.
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


def download_nemotron_model(
    target_dir="./models",
    repo_id="phosai/nemotron-3.5-asr-streaming-luganda",
    filename="nemotron-asr-streaming-luganda.nemo",
    token=None,
):
    os.makedirs(target_dir, exist_ok=True)
    hf_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    print(f"Downloading {filename} from {repo_id}...")
    if hf_token:
        print("Using Hugging Face token for authentication.")
    else:
        print("Note: No HF_TOKEN provided. If the repository is private, pass HF_TOKEN as an environment variable or argument.")

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
        description="Download PHOSAI Nemotron 3.5 Streaming Luganda ASR model"
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save the downloaded .nemo file (default: ./models)",
    )
    parser.add_argument(
        "--repo-id",
        default="phosai/nemotron-3.5-asr-streaming-luganda",
        help="Hugging Face repository ID (default: phosai/nemotron-3.5-asr-streaming-luganda)",
    )
    parser.add_argument(
        "--filename",
        default="nemotron-asr-streaming-luganda.nemo",
        help="Checkpoint filename inside repo (default: nemotron-asr-streaming-luganda.nemo)",
    )
    parser.add_argument(
        "--token",
        "--hf-token",
        dest="token",
        default=None,
        help="Hugging Face API token for private repos (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()
    download_nemotron_model(
        target_dir=args.output_dir,
        repo_id=args.repo_id,
        filename=args.filename,
        token=args.token,
    )
