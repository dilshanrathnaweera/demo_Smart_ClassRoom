"""
Download a sample ONNX model for local testing.
This script downloads a small SqueezeNet model from the ONNX model zoo and saves it as `model.onnx` in the repository root.

Usage:
    python scripts/download_sample_model.py

The repo intentionally excludes ONNX binaries via .gitignore; this helper script creates the file locally for development.
"""
from pathlib import Path
import urllib.request
import sys

MODEL_URL = "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
OUT = Path(__file__).resolve().parents[1] / 'model.onnx'

def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample ONNX model from {url} to {out_path}")
    try:
        with urllib.request.urlopen(url) as resp, open(out_path, 'wb') as f:
            chunk_size = 8192
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(2)

if __name__ == '__main__':
    download(MODEL_URL, OUT)
