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

A lightweight FastAPI application for detecting oil spills in satellite imagery using deep learning models.

## Features

- **U-Net Model**: Semantic segmentation for oil spill detection
- **DeepLab V3+ Model**: Advanced semantic segmentation
- **Lazy Loading**: Models load on-demand to save memory
- **RESTful API**: Easy integration with frontend applications

## Usage

### Health Check
```bash
GET /health
```

### Predict Oil Spill
```bash
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPG, PNG)
- model_choice: "model1" (U-Net) or "model2" (DeepLab)
```

### API Documentation
Visit `/docs` for interactive API documentation.

## Model Information

- **Model 1**: U-Net architecture for semantic segmentation
- **Model 2**: DeepLab V3+ for enhanced accuracy
- **Input Size**: 256x256 pixels
- **Output**: Binary classification (Oil Spill / No Oil Spill)

This API is optimized for HuggingFace Spaces with memory-efficient model loading.
