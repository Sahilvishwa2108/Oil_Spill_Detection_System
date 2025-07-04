# Oil Spill Detection System

Advanced AI-powered oil spill detection using ensemble machine learning models with consistent data handling and prediction formats.

## Features
- 🤖 **Dual AI Models**: U-Net (93.56% F1) + DeepLabV3+ (96.68% F1)
- 🔬 **Ensemble Prediction**: Combined analysis for optimal accuracy
- 🌐 **Full-Stack Solution**: Next.js frontend + FastAPI backend
- 📊 **Real-time Dashboard**: Interactive results visualization with consistent metrics
- 🚀 **Production Ready**: Deployed on Vercel + Render with optimized performance
- ✅ **Consistent Data**: Unified prediction formats across all components

## Recent Improvements
- ✅ **Resolved Data Inconsistencies**: Standardized model names, prediction formats, and confidence calculations
- ✅ **Backend-Frontend Alignment**: Consistent thresholds and class definitions
- ✅ **Enhanced Type Safety**: Updated TypeScript types to match actual API responses
- ✅ **Improved Error Handling**: Better fallback mechanisms and user feedback

## Quick Start

### Frontend
```bash
npm install
npm run dev
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

## Model Information
- **U-Net**: Lightweight segmentation (93.56% F1, 94.45% accuracy, 2.1M parameters)
- **DeepLabV3+**: High-accuracy segmentation (96.68% F1, 97.23% accuracy, 41.3M parameters)
- **Ensemble**: Weighted combination using F1 scores for optimal performance

## Prediction Format
All predictions now consistently return:
- **Binary Decision**: "Oil Spill Detected" or "No Oil Spill"
- **Confidence Score**: 0.0 to 1.0 (with consistent calculation methods)
- **Processing Time**: Actual model inference time
- **Prediction Mask**: Colored segmentation visualization

## Live Demo
- **Frontend**: [Live Dashboard](https://oil-spill-detection-system.vercel.app)
- **Backend**: [API Docs](https://your-render-app.onrender.com/docs)

## Models
- **UNet**: https://huggingface.co/sahilvishwa2108/oil-spill-unet
- **DeepLabV3+**: https://huggingface.co/sahilvishwa2108/oil-spill-deeplab

## Tech Stack
- **Frontend**: Next.js 15, TypeScript, Tailwind CSS
- **Backend**: FastAPI, TensorFlow, Python
- **Deployment**: Vercel, Render, Hugging Face
- **AI Models**: Computer Vision, Semantic Segmentation

---
*Built for portfolio showcase and environmental monitoring*
