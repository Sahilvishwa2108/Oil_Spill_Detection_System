---
title: Oil Spill Detection API
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Oil Spill Detection API

Advanced machine learning API for detecting oil spills in satellite and aerial imagery using state-of-the-art deep learning models (U-Net and DeepLab).

## Features
- 🧠 **Dual ML Models**: U-Net and DeepLab for semantic segmentation
- 📷 **Image Processing**: Automated preprocessing and postprocessing
- 🚀 **FastAPI Backend**: RESTful API with automatic documentation
- 🔍 **Multi-class Detection**: Oil spills, look-alikes, ships, and land
- 📊 **Confidence Scores**: Prediction reliability metrics
- ⚡ **Batch Processing**: Multiple image analysis

## Live Demo
🌐 **Frontend Dashboard**: [Oil Spill Detection Dashboard](https://oil-spill-frontend-oigeradm3-sahil-vishwakarmas-projects.vercel.app)

## API Usage

### Health Check
```bash
curl https://your-space-name.hf.space/health
```

### Predict Oil Spill
```bash
curl -X POST "https://your-space-name.hf.space/predict" \
  -F "file=@your_image.jpg" \
  -F "model_choice=model1"
```

### API Documentation
Visit `/docs` endpoint for interactive API documentation.

## Model Information
- **Model 1**: DeepLab V3+ for semantic segmentation
- **Model 2**: U-Net for oil spill detection
- **Input**: 256x256 RGB images
- **Output**: Multi-class segmentation masks

## Technologies
- FastAPI
- TensorFlow/Keras
- OpenCV
- Pillow
- NumPy
- Docker
