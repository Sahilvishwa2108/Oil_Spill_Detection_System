# Model Management Guide

## Why Models Aren't in Git

The ML model files (`.h5` files) are **NOT included in this repository** because:

- 🚫 **Size**: Models are 200MB+ each, exceeding GitHub's 100MB limit
- 💰 **Cost**: Git LFS charges for bandwidth on every download
- 🔄 **Performance**: Large files slow down git operations
- 🏗️ **Best Practice**: Models should be managed separately from code

## Model Storage Options

### Option 1: Cloud Storage (Recommended)
```bash
# Upload to AWS S3
aws s3 cp backend/models/deeplab_final_model.h5 s3://your-bucket/models/

# Upload to Google Cloud Storage
gsutil cp backend/models/deeplab_final_model.h5 gs://your-bucket/models/
```

### Option 2: Model Registry
- **Hugging Face**: Free hosting for ML models
- **MLflow**: Enterprise model registry
- **Weights & Biases**: Experiment tracking + model storage

### Option 3: Direct Download
Store models on any cloud service and download during deployment.

## Development Setup

For local development, you need to download the models manually:

1. **Download models** (get URLs from project maintainer):
   ```bash
   python download_models.py
   ```

2. **Or place models manually**:
   ```
   backend/models/
   ├── deeplab_final_model.h5
   └── unet_final_model.h5
   ```

## Production Deployment

### Docker Deployment
Models are downloaded automatically during Docker build:

```dockerfile
# In backend/Dockerfile
RUN python ../download_models.py
```

### Manual Deployment
```bash
# 1. Clone repository
git clone https://github.com/Sahilvishwa2108/Oil_Spill_Detection_System.git

# 2. Download models
python download_models.py

# 3. Start services
docker-compose up
```

## Model URLs Configuration

Update `download_models.py` with your actual model storage URLs:

```python
MODEL_URLS = {
    "deeplab_final_model.h5": "https://your-storage-url.com/deeplab_final_model.h5",
    "unet_final_model.h5": "https://your-storage-url.com/unet_final_model.h5"
}
```

## Security Notes

- 🔐 Use signed URLs for private models
- 🕐 Consider URL expiration for security
- 🔑 Store sensitive URLs as environment variables
