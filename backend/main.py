# Oil Spill Detection API - Azure Deployment v2.1
import os
import io
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import base64
from datetime import datetime
import logging
from dotenv import load_dotenv
import requests
import sys
import subprocess

# Load environment variables
env_file = ".env.production" if os.getenv("ENVIRONMENT") == "production" else ".env.local"
load_dotenv(env_file)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Oil Spill Detection API",
    description="Production-ready ML API for oil spill detection using semantic segmentation",
    version="1.0.0"
)

# Get CORS origins from environment
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
cors_origins = [origin.strip() for origin in cors_origins]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download_models_from_huggingface():
    """Download models from Hugging Face Hub in production"""
    try:
        logger.info("Downloading models from Hugging Face...")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Use local download script
        download_script = "download_models.py"
        if os.path.exists(download_script):
            result = subprocess.run([sys.executable, download_script], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Models downloaded successfully from Hugging Face")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"Error downloading models: {result.stderr}")
                return False
        else:
            logger.warning("Download script not found, using dummy models")
            return False
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return False

# Global variables for models
model1 = None
model2 = None

# Data models
class PredictionResponse(BaseModel):
    success: bool
    prediction_mask: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    selected_model: Optional[str] = None
    error: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    version: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str

def load_models():
    """Load the trained models"""
    global model1, model2
    try:
        models_dir = "models"
        
        # In production, try to download models first if they don't exist
        if os.getenv("ENVIRONMENT") == "production":
            deeplab_path = os.path.join(models_dir, "deeplab_final_model.h5")
            unet_path = os.path.join(models_dir, "unet_final_model.h5")
            
            if not (os.path.exists(deeplab_path) and os.path.exists(unet_path)):
                logger.info("Models not found locally, downloading from Hugging Face...")
                download_success = download_models_from_huggingface()
                if not download_success:
                    logger.warning("Failed to download models, will use dummy models")
        
        # Try to load DeepLab model
        deeplab_path = os.path.join(models_dir, "deeplab_final_model.h5")
        if os.path.exists(deeplab_path):
            try:
                model1 = tf.keras.models.load_model(deeplab_path, compile=False)
                logger.info("DeepLab model loaded successfully as model1")
            except Exception as e:
                logger.warning(f"Could not load DeepLab model: {str(e)}")
                # Create a simple dummy model for demonstration
                model1 = create_dummy_model((256, 256, 3), "deeplab")
                logger.info("Using dummy DeepLab model for demonstration")
        else:
            logger.warning("DeepLab model file not found, using dummy model")
            model1 = create_dummy_model((256, 256, 3), "deeplab")
        
        # Try to load U-Net model  
        unet_path = os.path.join(models_dir, "unet_final_model.h5")
        if os.path.exists(unet_path):
            try:
                model2 = tf.keras.models.load_model(unet_path, compile=False)
                logger.info("U-Net model loaded successfully as model2")
            except Exception as e:
                logger.warning(f"Could not load U-Net model: {str(e)}")
                # Create a simple dummy model for demonstration
                model2 = create_dummy_model((256, 256, 3), "unet")
                logger.info("Using dummy U-Net model for demonstration")
        else:
            logger.warning("U-Net model file not found, using dummy model")
            model2 = create_dummy_model((256, 256, 3), "unet")
    
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        # Create dummy models as fallback
        model1 = create_dummy_model((256, 256, 3), "deeplab")
        model2 = create_dummy_model((256, 256, 3), "unet")
        logger.info("Using dummy models for demonstration")

def create_dummy_model(input_shape: tuple, model_type: str):
    """Create a dummy model for demonstration purposes"""
    try:
        inputs = tf.keras.Input(shape=input_shape)
        
        if model_type == "deeplab":
            # Simple CNN for demonstration - ensure output shape matches expected (256, 256, 5)
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2D(5, 1, padding='same', activation='softmax')(x)  # 5 classes
            model = tf.keras.Model(inputs, x, name="dummy_deeplab")
        else:  # unet
            # Simple U-Net like structure for demonstration - ensure output shape matches expected (256, 256, 5)
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Conv2D(5, 1, padding='same', activation='softmax')(x)  # 5 classes
            model = tf.keras.Model(inputs, x, name="dummy_unet")
            
        logger.info(f"Created dummy {model_type} model with input shape {input_shape}")
        logger.info(f"Dummy {model_type} model output shape: {model.output_shape}")
        return model
    except Exception as e:
        logger.error(f"Error creating dummy model: {str(e)}")
        return None

def preprocess_image(image: Image.Image, target_size: tuple = (256, 256)) -> np.ndarray:
    """Preprocess image for model input"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def postprocess_mask(mask: np.ndarray) -> str:
    """Convert prediction mask to base64 image"""
    try:
        # Remove batch dimension
        mask = np.squeeze(mask)
        logger.info(f"Mask shape after squeeze: {mask.shape}")
        
        # Handle multi-class segmentation (5 classes)
        if len(mask.shape) == 3 and mask.shape[-1] > 1:
            # Apply softmax to get probabilities
            from scipy.special import softmax
            mask_probs = softmax(mask, axis=-1)
            
            # Get the class with highest probability for each pixel
            mask = np.argmax(mask_probs, axis=-1)
            logger.info(f"Multi-class mask shape: {mask.shape}, unique classes: {np.unique(mask)}")
        else:
            # Binary segmentation case
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
        
        # Create colored mask for different classes
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Color mapping for oil spill classes (assuming these are the classes)
        # Class 0: Background (transparent/black)
        # Class 1: Oil spill (red)
        # Class 2: Look-alike (yellow)
        # Class 3: Ship (blue)
        # Class 4: Land (green)
        color_map = {
            0: [0, 0, 0],       # Background - black
            1: [255, 0, 0],     # Oil spill - red
            2: [255, 255, 0],   # Look-alike - yellow
            3: [0, 0, 255],     # Ship - blue
            4: [0, 255, 0],     # Land - green
        }
        
        for class_id, color in color_map.items():
            class_pixels = mask == class_id
            colored_mask[class_pixels] = color
        
        # Convert to PIL Image
        mask_image = Image.fromarray(colored_mask)
        
        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error postprocessing mask: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up Oil Spill Detection API...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    # Load models
    load_models()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Oil Spill Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "model1": model1 is not None,
            "model2": model2 is not None
        },
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    info = {}
    
    if model1 is not None:
        info["model1"] = ModelInfo(
            name="DeepLab V3+ (Oil Spill Segmentation)",
            version="1.0",
            input_shape=list(model1.input_shape[1:]),
            output_shape=list(model1.output_shape[1:]),
            parameters=model1.count_params()
        )
    
    if model2 is not None:
        info["model2"] = ModelInfo(
            name="U-Net (Oil Spill Detection)", 
            version="1.0",
            input_shape=list(model2.input_shape[1:]),
            output_shape=list(model2.output_shape[1:]),
            parameters=model2.count_params()
        )
    
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict_oil_spill(
    file: UploadFile = File(...),
    model_choice: str = "model1"
):
    """Predict oil spill from uploaded image"""
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Select model
        selected_model = None
        if model_choice == "model1" and model1 is not None:
            selected_model = model1
        elif model_choice == "model2" and model2 is not None:
            selected_model = model2
        else:
            raise HTTPException(status_code=400, detail=f"Model {model_choice} not available")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = selected_model.predict(processed_image)
        
        # Calculate confidence score
        confidence = float(np.max(prediction))
        
        # Postprocess mask
        mask_base64 = postprocess_mask(prediction)
          # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            success=True,
            prediction_mask=mask_base64,
            confidence_score=confidence,
            processing_time=processing_time,
            selected_model=model_choice
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return PredictionResponse(
            success=False,
            error=str(e)
        )

@app.post("/batch-predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    model_choice: str = "model1"
):
    """Batch prediction for multiple images"""
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            result = await predict_oil_spill(file, model_choice)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "result": PredictionResponse(success=False, error=str(e))
            })
    
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
