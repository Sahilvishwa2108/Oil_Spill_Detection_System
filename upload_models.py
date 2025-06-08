#!/usr/bin/env python3
"""
Upload Oil Spill Detection Models to Hugging Face Hub
Run this script to upload your trained models to Hugging Face
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import getpass

def upload_models():
    """Upload models to Hugging Face Hub"""
    
    # Get user token
    print("🤗 Oil Spill Detection - Model Uploader")
    print("=" * 50)
    
    token = getpass.getpass("Enter your Hugging Face token (input will be hidden): ")
    
    if not token:
        print("❌ No token provided. Exiting.")
        return
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Repository details
    repo_id = "sahilvishwa2108/oil-spill-detection-models"
    
    try:
        # Create repository
        print(f"📦 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository created successfully!")
        
        # Model files to upload
        models_dir = Path("backend/models")
        model_files = [
            "deeplab_final_model.h5",
            "unet_final_model.h5"
        ]
        
        # Upload each model
        for model_file in model_files:
            model_path = models_dir / model_file
            
            if not model_path.exists():
                print(f"⚠️  Model file not found: {model_path}")
                continue
            
            print(f"⬆️  Uploading {model_file}...")
            
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_file,
                repo_id=repo_id,
                token=token,
                repo_type="model"
            )
            
            print(f"✅ Successfully uploaded {model_file}")
        
        # Create README for the model repository
        readme_content = """---
library_name: tensorflow
tags:
- computer-vision
- oil-spill-detection
- semantic-segmentation
- environmental-monitoring
license: mit
---

# Oil Spill Detection Models

This repository contains trained deep learning models for oil spill detection in satellite and aerial imagery.

## Models Included

- **deeplab_final_model.h5**: DeepLab V3+ model for semantic segmentation
- **unet_final_model.h5**: U-Net model for biomedical image segmentation

## Usage

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('deeplab_final_model.h5')

# Make predictions
predictions = model.predict(your_image_data)
```

## Model Performance

- Input Shape: (256, 256, 3)
- Output: Multi-class segmentation masks
- Optimized for oil spill detection accuracy

## Citation

If you use these models in your research, please cite:

```
@misc{oil-spill-detection-2025,
  title={Oil Spill Detection Using Deep Learning},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/sahilvishwa2108/oil-spill-detection-models}
}
```
"""
        
        # Upload README
        with open("temp_readme.md", "w") as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj="temp_readme.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type="model"
        )
        
        # Clean up temp file
        os.remove("temp_readme.md")
        
        print("✅ README uploaded successfully!")
        print("\n🎉 All models uploaded successfully!")
        print(f"🌐 View your models at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        print("\n💡 Tips:")
        print("1. Check your token has 'Write' permissions")
        print("2. Make sure you're connected to the internet")
        print("3. Verify the model files exist in backend/models/")

if __name__ == "__main__":
    upload_models()
