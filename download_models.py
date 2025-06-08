#!/usr/bin/env python3
"""
Model Download Script for Oil Spill Detection System
Downloads pre-trained models from cloud storage during deployment
"""

import os
import requests
from pathlib import Path
import sys

# Hugging Face model repository
HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "sahilvishwa2108/oil-spill-detection-models")

# Model files to download from Hugging Face
MODEL_FILES = {
    "deeplab_final_model.h5": "deeplab_final_model.h5",
    "unet_final_model.h5": "unet_final_model.h5"
}

def download_model(repo: str, filename: str, models_dir: Path) -> bool:
    """Download a model file from Hugging Face Hub"""
    try:
        # Construct Hugging Face download URL
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        
        print(f"Downloading {filename} from Hugging Face...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        model_path = models_dir / filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    """Download all required models"""
    script_dir = Path(__file__).parent
    models_dir = script_dir / "backend" / "models"
    
    print("🤖 Oil Spill Detection - Model Downloader")
    print("=" * 50)
      # Check if models already exist
    all_exist = all((models_dir / filename).exists() for filename in MODEL_FILES.keys())
    if all_exist:
        print("✅ All models already downloaded!")
        return
    
    # Download missing models
    success_count = 0
    for local_filename, remote_filename in MODEL_FILES.items():
        model_path = models_dir / local_filename
        if model_path.exists():
            print(f"⏭️  {local_filename} already exists, skipping...")
            success_count += 1
            continue
            
        if download_model(HUGGINGFACE_REPO, remote_filename, models_dir):
            success_count += 1
    
    if success_count == len(MODEL_FILES):
        print(f"\n🎉 Successfully downloaded all {len(MODEL_FILES)} models!")
    else:
        print(f"\n⚠️  Downloaded {success_count}/{len(MODEL_FILES)} models")
        sys.exit(1)

if __name__ == "__main__":
    main()
