#!/usr/bin/env python3
"""
Model Download Script for Oil Spill Detection System
Downloads pre-trained models from cloud storage during deployment
"""

import os
import requests
from pathlib import Path
import sys

# Model URLs (replace with your actual model storage URLs)
MODEL_URLS = {
    "deeplab_final_model.h5": "https://your-cloud-storage.com/models/deeplab_final_model.h5",
    "unet_final_model.h5": "https://your-cloud-storage.com/models/unet_final_model.h5"
}

def download_model(url: str, filename: str, models_dir: Path) -> bool:
    """Download a model file from URL"""
    try:
        print(f"Downloading {filename}...")
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
    all_exist = all((models_dir / filename).exists() for filename in MODEL_URLS.keys())
    if all_exist:
        print("✅ All models already downloaded!")
        return
    
    # Download missing models
    success_count = 0
    for filename, url in MODEL_URLS.items():
        model_path = models_dir / filename
        if model_path.exists():
            print(f"⏭️  {filename} already exists, skipping...")
            success_count += 1
            continue
            
        if download_model(url, filename, models_dir):
            success_count += 1
    
    if success_count == len(MODEL_URLS):
        print(f"\n🎉 Successfully downloaded all {len(MODEL_URLS)} models!")
    else:
        print(f"\n⚠️  Downloaded {success_count}/{len(MODEL_URLS)} models")
        sys.exit(1)

if __name__ == "__main__":
    main()
