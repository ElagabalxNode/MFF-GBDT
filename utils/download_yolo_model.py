"""
Utility script to download YOLO models manually when automatic download fails.
This helps bypass network/SSL issues with Ultralytics automatic download.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Model URLs from Ultralytics assets repository
MODEL_URLS = {
    "yolo11n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    "yolo11s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
    "yolo11m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
    "yolo11l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
    "yolo11x-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
    "yolov8s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
    "yolov8m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt",
    "yolov8l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-seg.pt",
    "yolov8x-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-seg.pt",
}

def download_file(url: str, filepath: str, chunk_size: int = 8192):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/download_yolo_model.py <model_name> [output_dir]")
        print("\nAvailable models:")
        for model_name in sorted(MODEL_URLS.keys()):
            print(f"  - {model_name}")
        sys.exit(1)
    
    model_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    
    if model_name not in MODEL_URLS:
        print(f"Error: Unknown model '{model_name}'")
        print("Available models:", ", ".join(MODEL_URLS.keys()))
        sys.exit(1)
    
    url = MODEL_URLS[model_name]
    output_path = os.path.join(output_dir, model_name)
    
    # Check if file already exists
    if os.path.exists(output_path):
        response = input(f"File {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)
    
    print(f"Downloading {model_name} from {url}")
    print(f"Output: {output_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if download_file(url, output_path):
        print(f"\n[SUCCESS] Successfully downloaded {model_name} to {output_path}")
    else:
        print(f"\n[ERROR] Failed to download {model_name}")
        print("You can try:")
        print("1. Check your internet connection")
        print("2. Download manually from:", url)
        print("3. Place the file in:", output_dir)
        sys.exit(1)

if __name__ == "__main__":
    main()

