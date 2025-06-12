"""
Oil Spill Detection API - Optimized for Render Deployment
Downloads models on startup and loads them on-demand to reduce memory usage
"""

import os
import io
import numpy as np
import requests
from pathlib import Path
from PIL import Image
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from datetime import datetime
import gc

# Configure environment for optimal performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Model download configuration
HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "sahilvishwa2108/oil-spill-detection-models")
MODEL_FILES = {
    "unet_final_model.h5": "unet_final_model.h5",
    "deeplab_final_model.h5": "deeplab_final_model.h5"
}

def download_model_if_needed(filename: str) -> bool:
    """Download model from HuggingFace if not exists locally"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / filename
    
    if model_path.exists():
        print(f"✅ Model {filename} already exists")
        return True
    
    try:
        url = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main/{filename}"
        print(f"⬇️ Downloading {filename} from HuggingFace...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

# Download models on startup
print("🤖 Initializing Oil Spill Detection API...")
for filename in MODEL_FILES.values():
    download_model_if_needed(filename)

app = FastAPI(
    title="Oil Spill Detection API",
    description="Lightweight API for oil spill detection using deep learning",
    version="1.0.0"
)

# CORS middleware for frontend access
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://oil-spill-detection-system.vercel.app",  # Your actual Vercel domain
    "https://oil-spill-frontend-oigeradm3-sahil-vishwakarmas-projects.vercel.app",
    "*"  # Temporarily allow all origins for debugging
]

# Get origins from environment variable if available
env_origins = os.getenv("CORS_ORIGINS")
if env_origins:
    allowed_origins.extend(env_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
model1 = None
model2 = None

class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: dict

def lazy_load_model1():
    """Load model1 only when needed"""
    global model1
    if model1 is None:
        try:
            import tensorflow as tf
            model_path = "models/unet_final_model.h5"
            if os.path.exists(model_path):
                model1 = tf.keras.models.load_model(model_path)
                print("Model 1 (U-Net) loaded successfully")
            else:
                print(f"Model 1 not found at {model_path}")
        except Exception as e:
            print(f"Error loading model 1: {e}")
    return model1

def lazy_load_model2():
    """Load model2 only when needed"""
    global model2
    if model2 is None:
        try:
            import tensorflow as tf
            model_path = "models/deeplab_final_model.h5"
            if os.path.exists(model_path):
                model2 = tf.keras.models.load_model(model_path)
                print("Model 2 (DeepLab) loaded successfully")
            else:
                print(f"Model 2 not found at {model_path}")
        except Exception as e:
            print(f"Error loading model 2: {e}")
    return model2

def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_oil_spill(image_array, model):
    """Make prediction using the specified model"""
    try:
        # Make prediction
        prediction = model.predict(image_array)
        
        # Process prediction (assuming binary classification)
        confidence = float(np.max(prediction))
        predicted_class = "Oil Spill Detected" if confidence > 0.5 else "No Oil Spill"
        
        return predicted_class, confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check if model files exist (they may not be loaded in memory yet due to lazy loading)
    model1_exists = os.path.exists("models/unet_final_model.h5")
    model2_exists = os.path.exists("models/deeplab_final_model.h5")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "model1": model1_exists,
            "model2": model2_exists
        }
    )

@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    # Check if model files exist
    model1_exists = os.path.exists("models/unet_final_model.h5")
    model2_exists = os.path.exists("models/deeplab_final_model.h5")
    
    return {
        "models": {
            "model1": {
                "name": "U-Net",
                "description": "U-Net model for semantic segmentation",
                "loaded": model1_exists
            },
            "model2": {
                "name": "DeepLab V3+",
                "description": "DeepLab V3+ model for semantic segmentation", 
                "loaded": model2_exists
            }
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_choice: str = "model1"
):
    """Predict oil spill in uploaded image"""
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Load appropriate model
        if model_choice == "model1":
            model = lazy_load_model1()
            model_name = "U-Net"
        else:
            model = lazy_load_model2()
            model_name = "DeepLab V3+"
        
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail=f"Model {model_choice} is not available"
            )
        
        # Make prediction
        predicted_class, confidence = predict_oil_spill(processed_image, model)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up memory
        gc.collect()
        
        return PredictionResponse(
            success=True,
            prediction=predicted_class,
            confidence=round(confidence, 4),
            processing_time=round(processing_time, 2),
            model_used=model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return PredictionResponse(
            success=False,
            error=str(e)
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Oil Spill Detection API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/models/info", 
            "predict": "/predict",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
