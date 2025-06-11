#!/usr/bin/env python3
"""
Model Download Script for Oil Spill Detection System
Downloads pre-trained models from Hugging Face Hub during deployment
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
    models_dir = script_dir / "models"
    
    print("🤖 Oil Spill Detection - Model Downloader")
    print("=" * 50)
    
    # Check if models already exist
    existing_models = []
    missing_models = []
    
    for local_name, remote_name in MODEL_FILES.items():
        model_path = models_dir / local_name
        if model_path.exists():
            existing_models.append(local_name)
            print(f"✅ {local_name} already exists")
        else:
            missing_models.append((local_name, remote_name))
    
    if not missing_models:
        print("\n🎉 All models are already downloaded!")
        return True
    
    # Download missing models
    print(f"\n📥 Downloading {len(missing_models)} missing models...")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for local_name, remote_name in missing_models:
        if download_model(HUGGINGFACE_REPO, remote_name, models_dir):
            success_count += 1
    
    if success_count == len(missing_models):
        print(f"\n🎉 Successfully downloaded all {success_count} models!")
        return True
    else:
        print(f"\n⚠️  Downloaded {success_count}/{len(missing_models)} models")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
