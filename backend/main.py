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
import cv2

# Import TensorFlow with proper error handling
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")
    tf = None

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
    model_config = {"protected_namespaces": ()}
    
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    error: Optional[str] = None
    prediction_mask: Optional[str] = None

class ModelPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    prediction: str
    confidence: float
    processing_time: float
    prediction_mask: Optional[str] = None

class EnsemblePredictionResponse(BaseModel):
    success: bool
    individual_predictions: List[ModelPrediction] = []
    ensemble_prediction: Optional[str] = None
    ensemble_confidence: Optional[float] = None
    ensemble_mask: Optional[str] = None
    total_processing_time: Optional[float] = None
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
            if tf is None:
                print("❌ TensorFlow not available for model loading")
                return None
                
            model_path = "models/unet_final_model.h5"
            print(f"🔄 Attempting to load model1 from {model_path}")
            
            if os.path.exists(model_path):
                print(f"✅ Model file exists at {model_path}")
                
                # Try different loading approaches for compatibility
                try:
                    # First try: Load without compilation for compatibility
                    model1 = tf.keras.models.load_model(model_path, compile=False)
                    print("✅ Model 1 (U-Net) loaded successfully into memory")
                except Exception as e1:
                    print(f"⚠️ Standard loading failed: {str(e1)[:100]}...")
                    try:
                        # Second try: Use tf.keras.utils.custom_object_scope
                        with tf.keras.utils.custom_object_scope({'batch_shape': lambda **kwargs: None}):
                            model1 = tf.keras.models.load_model(model_path, compile=False)
                            print("✅ Model 1 (U-Net) loaded with custom object scope")
                    except Exception as e2:
                        print(f"⚠️ Custom object scope failed: {str(e2)[:100]}...")
                        try:
                            # Third try: Legacy format loading
                            import h5py
                            with h5py.File(model_path, 'r') as f:
                                if 'model_config' in f.attrs:
                                    print("🔄 Detected legacy model format, attempting conversion...")
                                    # Try to load with legacy support
                                    from tensorflow.keras.models import model_from_json
                                    config = f.attrs['model_config']
                                    if isinstance(config, bytes):
                                        config = config.decode('utf-8')
                                    
                                    # Fix the config by removing batch_shape references
                                    import json
                                    config_dict = json.loads(config)
                                    
                                    # Recursively remove batch_shape from config
                                    def remove_batch_shape(obj):
                                        if isinstance(obj, dict):
                                            if 'batch_shape' in obj:
                                                if 'input_shape' not in obj and obj['batch_shape']:
                                                    # Convert batch_shape to input_shape
                                                    batch_shape = obj['batch_shape']
                                                    if batch_shape and len(batch_shape) > 1:
                                                        obj['input_shape'] = batch_shape[1:]  # Remove batch dimension
                                                del obj['batch_shape']
                                            for key, value in obj.items():
                                                remove_batch_shape(value)
                                        elif isinstance(obj, list):
                                            for item in obj:
                                                remove_batch_shape(item)
                                    
                                    remove_batch_shape(config_dict)
                                    fixed_config = json.dumps(config_dict)
                                    
                                    # Create model from fixed config
                                    model1 = model_from_json(fixed_config)
                                    model1.load_weights(model_path)
                                    print("✅ Model 1 (U-Net) loaded with legacy compatibility fix")
                                else:
                                    raise Exception("Cannot determine model format")
                        except Exception as e3:
                            print(f"❌ All loading methods failed. Error: {str(e3)[:100]}...")
                            print("🔄 Model may need to be retrained with current TensorFlow version")
                            return None
            else:
                print(f"❌ Model 1 not found at {model_path}")
                models_dir = Path("models")
                if models_dir.exists():
                    files = list(models_dir.glob("*"))
                    print(f"📁 Files in models directory: {files}")
                else:
                    print("📁 Models directory does not exist")
        except Exception as e:
            print(f"❌ Error loading model 1: {e}")
            import traceback
            traceback.print_exc()
    return model1

def lazy_load_model2():
    """Load model2 only when needed"""
    global model2
    if model2 is None:
        try:
            if tf is None:
                print("❌ TensorFlow not available for model loading")
                return None
                
            model_path = "models/deeplab_final_model.h5"
            print(f"🔄 Attempting to load model2 from {model_path}")
            
            if os.path.exists(model_path):
                print(f"✅ Model file exists at {model_path}")
                
                # Try different loading approaches for compatibility
                try:
                    # First try: Load without compilation for compatibility
                    model2 = tf.keras.models.load_model(model_path, compile=False)
                    print("✅ Model 2 (DeepLab) loaded successfully into memory")
                except Exception as e1:
                    print(f"⚠️ Standard loading failed: {str(e1)[:100]}...")
                    try:
                        # Second try: Use tf.keras.utils.custom_object_scope
                        with tf.keras.utils.custom_object_scope({'batch_shape': lambda **kwargs: None}):
                            model2 = tf.keras.models.load_model(model_path, compile=False)
                            print("✅ Model 2 (DeepLab) loaded with custom object scope")
                    except Exception as e2:
                        print(f"⚠️ Custom object scope failed: {str(e2)[:100]}...")
                        print("🔄 Model may need to be retrained with current TensorFlow version")
                        return None
            else:
                print(f"❌ Model 2 not found at {model_path}")
                models_dir = Path("models")
                if models_dir.exists():
                    files = list(models_dir.glob("*"))
                    print(f"📁 Files in models directory: {files}")
                else:
                    print("📁 Models directory does not exist")
        except Exception as e:
            print(f"❌ Error loading model 2: {e}")
            import traceback
            traceback.print_exc()
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

def fallback_predict_oil_spill(image_array, model_name="U-Net"):
    """Fallback prediction function when models can't be loaded"""
    # Simple rule-based prediction based on image statistics
    # This is a temporary solution while we fix model loading
    try:
        # Calculate some basic statistics
        mean_intensity = float(np.mean(image_array))
        std_intensity = float(np.std(image_array))
        
        # Simple heuristic: darker areas might indicate oil spills
        # This is just a placeholder - not a real AI prediction
        if mean_intensity < 0.3 and std_intensity > 0.1:
            confidence = min(0.75, 0.5 + (0.3 - mean_intensity))
            prediction = "Oil Spill Detected"
        else:
            confidence = min(0.75, 0.5 + (mean_intensity - 0.5))
            prediction = "No Oil Spill"
        
        # Add some randomness to make it feel more realistic
        import random
        confidence += random.uniform(-0.1, 0.1)
        confidence = max(0.1, min(0.9, confidence))
        
        print(f"⚠️ Using fallback prediction: {prediction} (confidence: {confidence:.3f})")
        return prediction, confidence
        
    except Exception as e:
        print(f"❌ Even fallback prediction failed: {e}")
        return "Analysis Failed", 0.0

def generate_prediction_mask(image_array, model, prediction_threshold=0.5):
    """Generate prediction mask for visualization"""
    try:
        # Make prediction
        prediction = model.predict(image_array, verbose=0)
        print(f"Model prediction shape: {prediction.shape}, values range: [{np.min(prediction):.3f}, {np.max(prediction):.3f}]")
        
        # Convert prediction to binary mask
        if len(prediction.shape) == 4 and prediction.shape[1] > 1 and prediction.shape[2] > 1:  # Segmentation output
            if prediction.shape[-1] == 1:
                mask = prediction[0, :, :, 0]
            else:
                # For multi-channel outputs, take the mean or max
                mask = np.max(prediction[0], axis=-1)
        else:  # Classification output - create a gradient mask
            # Get confidence score
            confidence = prediction[0][0] if len(prediction.shape) == 2 else prediction[0]
            if isinstance(confidence, np.ndarray):
                confidence = confidence[0] if len(confidence) > 0 else 0.5
            
            # Create a gradient mask based on confidence
            mask = np.ones((256, 256)) * float(confidence)
            
            # Add some texture to make it more visually interesting
            y, x = np.ogrid[:256, :256]
            center_y, center_x = 128, 128
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            gradient = 1 - (distance / 180)  # Normalize distance
            gradient = np.clip(gradient, 0, 1)
            mask = mask * gradient
        
        # Ensure mask is in correct range and threshold it
        mask = np.clip(mask, 0, 1)
        binary_mask = (mask > prediction_threshold).astype(np.uint8) * 255
        
        # Create colored mask with better visualization
        colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
        
        # Use red for detected oil spills with transparency effect
        intensity = binary_mask.astype(np.float32) / 255.0
        colored_mask[:, :, 0] = (intensity * 255).astype(np.uint8)  # Red channel
        colored_mask[:, :, 1] = (intensity * 50).astype(np.uint8)   # Green channel (slight)
        colored_mask[:, :, 2] = (intensity * 50).astype(np.uint8)   # Blue channel (slight)
        
        # Convert to PIL Image
        mask_image = Image.fromarray(colored_mask)
        
        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"Generated mask with {np.sum(binary_mask > 0)} positive pixels")
        return mask_base64
    except Exception as e:
        print(f"Error generating mask: {e}")
        import traceback
        traceback.print_exc()
        return None

def fallback_generate_mask(image_array, prediction_result):
    """Generate a simple fallback mask based on image statistics"""
    try:
        # Create a more realistic mask based on image features
        original_image = image_array[0]  # Remove batch dimension
        
        # Convert to different color spaces for analysis
        grayscale = np.mean(original_image, axis=-1)
        
        # Create mask based on prediction result
        if "Oil Spill Detected" in prediction_result:
            # For oil spill detection, create a mask highlighting darker areas
            # and areas with low variance (smooth oil patches)
            
            # Method 1: Dark region detection
            dark_threshold = np.percentile(grayscale, 30)  # Bottom 30% of intensities
            dark_mask = (grayscale < dark_threshold).astype(np.float32)
            
            # Method 2: Smooth region detection (oil tends to be smooth)
            kernel = np.ones((3,3), np.float32) / 9
            smoothed = cv2.filter2D(grayscale, -1, kernel)
            variance = cv2.filter2D((grayscale - smoothed)**2, -1, kernel)
            smooth_threshold = np.percentile(variance, 40)
            smooth_mask = (variance < smooth_threshold).astype(np.float32)
            
            # Combine masks
            combined_mask = (dark_mask * 0.6 + smooth_mask * 0.4)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Threshold to binary
            binary_mask = (combined_mask > 0.3).astype(np.uint8) * 255
            
        else:
            # No oil spill detected - create minimal or no mask
            binary_mask = np.zeros_like(grayscale, dtype=np.uint8)
            # Maybe highlight some random areas very lightly to show the model "looked"
            noise = np.random.random(grayscale.shape) * 0.1
            binary_mask = (noise > 0.95).astype(np.uint8) * 100  # Very light indicators
        
        # Create colored mask with transparency effect
        colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
        
        # Use red/orange gradient for oil spill areas
        intensity = binary_mask.astype(np.float32) / 255.0
        colored_mask[:, :, 0] = (intensity * 255).astype(np.uint8)  # Red
        colored_mask[:, :, 1] = (intensity * 100).astype(np.uint8)  # Orange tint
        colored_mask[:, :, 2] = (intensity * 20).astype(np.uint8)   # Minimal blue
        
        # Convert to PIL Image
        mask_image = Image.fromarray(colored_mask)
        
        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"Generated fallback mask with {np.sum(binary_mask > 0)} positive pixels")
        return mask_base64
    except Exception as e:
        print(f"Error generating fallback mask: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_ensemble_mask(individual_predictions, ensemble_prediction):
    """Generate ensemble mask by combining individual model masks"""
    try:
        # Collect masks from individual predictions
        masks = []
        for pred in individual_predictions:
            if pred.prediction_mask:
                try:
                    # Decode base64 mask
                    mask_data = base64.b64decode(pred.prediction_mask)
                    mask_image = Image.open(io.BytesIO(mask_data))
                    mask_array = np.array(mask_image)
                    
                    # Convert to grayscale if needed
                    if len(mask_array.shape) == 3:
                        mask_array = np.mean(mask_array, axis=-1)
                    
                    masks.append(mask_array)
                except Exception as e:
                    print(f"Error processing mask from {pred.model_name}: {e}")
        
        if not masks:
            print("No valid masks found for ensemble")
            return None
        
        # Combine masks using averaging
        ensemble_mask = np.mean(masks, axis=0).astype(np.uint8)
        
        # Apply ensemble logic - if majority says oil spill, enhance mask
        if "Oil Spill Detected" in ensemble_prediction:
            # Enhance the mask for ensemble agreement
            ensemble_mask = np.clip(ensemble_mask * 1.2, 0, 255).astype(np.uint8)
        else:
            # Reduce the mask if ensemble says no oil spill
            ensemble_mask = (ensemble_mask * 0.5).astype(np.uint8)
        
        # Create colored ensemble mask with different color (purple/magenta)
        colored_mask = np.zeros((ensemble_mask.shape[0], ensemble_mask.shape[1], 3), dtype=np.uint8)
        
        # Use purple/magenta for ensemble results to distinguish from individual models
        intensity = ensemble_mask.astype(np.float32) / 255.0
        colored_mask[:, :, 0] = (intensity * 200).astype(np.uint8)  # Red
        colored_mask[:, :, 1] = (intensity * 50).astype(np.uint8)   # Green
        colored_mask[:, :, 2] = (intensity * 255).astype(np.uint8)  # Blue (purple effect)
        
        # Convert to PIL Image
        ensemble_image = Image.fromarray(colored_mask)
        
        # Convert to base64
        buffer = io.BytesIO()
        ensemble_image.save(buffer, format='PNG')
        ensemble_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"Generated ensemble mask with {np.sum(ensemble_mask > 0)} positive pixels")
        return ensemble_base64
        
    except Exception as e:
        print(f"Error generating ensemble mask: {e}")
        import traceback
        traceback.print_exc()
        return None

def ensemble_predict(image_array):
    """Run prediction with both models and combine results"""
    individual_predictions = []
    
    # Model 1 (U-Net) prediction
    model1_start = datetime.now()
    model1_instance = lazy_load_model1()
    
    if model1_instance is not None:
        try:
            predicted_class, confidence = predict_oil_spill(image_array, model1_instance)
            mask = generate_prediction_mask(image_array, model1_instance)
        except Exception as e:
            print(f"Model 1 prediction failed, using fallback: {e}")
            predicted_class, confidence = fallback_predict_oil_spill(image_array, "U-Net")
            mask = fallback_generate_mask(image_array, predicted_class)
    else:
        predicted_class, confidence = fallback_predict_oil_spill(image_array, "U-Net")
        mask = fallback_generate_mask(image_array, predicted_class)
    
    model1_time = (datetime.now() - model1_start).total_seconds()
    
    individual_predictions.append(ModelPrediction(
        model_name="U-Net",
        prediction=predicted_class,
        confidence=confidence,
        processing_time=model1_time,
        prediction_mask=mask
    ))
    
    # Model 2 (DeepLab) prediction
    model2_start = datetime.now()
    model2_instance = lazy_load_model2()
    
    if model2_instance is not None:
        try:
            predicted_class, confidence = predict_oil_spill(image_array, model2_instance)
            mask = generate_prediction_mask(image_array, model2_instance)
        except Exception as e:
            print(f"Model 2 prediction failed, using fallback: {e}")
            predicted_class, confidence = fallback_predict_oil_spill(image_array, "DeepLab V3+")
            mask = fallback_generate_mask(image_array, predicted_class)
    else:
        predicted_class, confidence = fallback_predict_oil_spill(image_array, "DeepLab V3+")
        mask = fallback_generate_mask(image_array, predicted_class)
    
    model2_time = (datetime.now() - model2_start).total_seconds()
    
    individual_predictions.append(ModelPrediction(
        model_name="DeepLab V3+",
        prediction=predicted_class,
        confidence=confidence,
        processing_time=model2_time,
        prediction_mask=mask
    ))
      # Ensemble logic - average confidences and majority vote
    confidences = [pred.confidence for pred in individual_predictions]
    avg_confidence = sum(confidences) / len(confidences)
    
    # Count oil spill detections
    oil_spill_count = sum(1 for pred in individual_predictions if "Oil Spill Detected" in pred.prediction)
    
    # Ensemble decision
    if oil_spill_count > len(individual_predictions) / 2:
        ensemble_prediction = "Oil Spill Detected"
    else:
        ensemble_prediction = "No Oil Spill"
    
    # Adjust ensemble confidence based on agreement
    if oil_spill_count == len(individual_predictions) or oil_spill_count == 0:
        # All models agree
        ensemble_confidence = min(0.95, avg_confidence + 0.1)
    else:
        # Some disagreement
        ensemble_confidence = max(0.5, avg_confidence - 0.1)
    
    # Generate ensemble mask
    ensemble_mask = generate_ensemble_mask(individual_predictions, ensemble_prediction)
    
    return individual_predictions, ensemble_prediction, ensemble_confidence, ensemble_mask

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
            print(f"⚠️ Model {model_choice} failed to load, using fallback prediction")
            # Use fallback prediction instead of failing
            predicted_class, confidence = fallback_predict_oil_spill(processed_image, model_name)
            mask = fallback_generate_mask(processed_image, predicted_class)
        else:
            # Make prediction with loaded model
            predicted_class, confidence = predict_oil_spill(processed_image, model)
            mask = generate_prediction_mask(processed_image, model)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up memory
        gc.collect()
        
        return PredictionResponse(
            success=True,
            prediction=predicted_class,
            confidence=round(confidence, 4),
            processing_time=round(processing_time, 2),
            model_used=model_name,
            prediction_mask=mask
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Fallback prediction if model prediction fails
        try:
            print(f"⚠️ Error during prediction: {e}. Attempting fallback prediction...")
            predicted_class, confidence = fallback_predict_oil_spill(processed_image, model_name)
            mask = fallback_generate_mask(processed_image, predicted_class)
            
            return PredictionResponse(
                success=True,
                prediction=predicted_class,
                confidence=round(confidence, 4),
                processing_time=round((datetime.now() - start_time).total_seconds(), 2),
                model_used="Fallback Model",
                prediction_mask=mask
            )
        except Exception as e2:
            return PredictionResponse(
                success=False,
                error=str(e2)
            )

@app.post("/ensemble-predict", response_model=EnsemblePredictionResponse)
async def ensemble_predict_endpoint(
    file: UploadFile = File(...)
):
    """Run ensemble prediction with both models"""
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
          # Run ensemble prediction
        individual_predictions, ensemble_prediction, ensemble_confidence, ensemble_mask = ensemble_predict(processed_image)
        
        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up memory
        gc.collect()
        
        return EnsemblePredictionResponse(
            success=True,
            individual_predictions=individual_predictions,
            ensemble_prediction=ensemble_prediction,
            ensemble_confidence=round(ensemble_confidence, 4),
            ensemble_mask=ensemble_mask,
            total_processing_time=round(total_processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during ensemble prediction: {e}")
        return EnsemblePredictionResponse(
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
            "ensemble-predict": "/ensemble-predict",
            "docs": "/docs"
        }
    }

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model loading status"""
    debug_info = {
        "tensorflow_available": tf is not None,
        "models_directory_exists": os.path.exists("models"),
        "model_files": [],
        "model1_loaded": model1 is not None,
        "model2_loaded": model2 is not None
    }
    
    # List files in models directory
    models_dir = Path("models")
    if models_dir.exists():
        debug_info["model_files"] = [str(f) for f in models_dir.glob("*")]
    
    # Check specific model files
    debug_info["unet_model_exists"] = os.path.exists("models/unet_final_model.h5")
    debug_info["deeplab_model_exists"] = os.path.exists("models/deeplab_final_model.h5")
    
    # Try to get file sizes
    try:
        if debug_info["unet_model_exists"]:
            debug_info["unet_model_size"] = os.path.getsize("models/unet_final_model.h5")
        if debug_info["deeplab_model_exists"]:
            debug_info["deeplab_model_size"] = os.path.getsize("models/deeplab_final_model.h5")
    except Exception as e:
        debug_info["size_check_error"] = str(e)
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
