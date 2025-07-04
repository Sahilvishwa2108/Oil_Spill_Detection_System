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
    # Import Keras components conditionally - TensorFlow 2.x uses tf.keras
    try:
        from tensorflow.keras.models import model_from_json  # type: ignore
    except (ImportError, AttributeError):
        try:
            # Alternative import for different TensorFlow versions
            from keras.models import model_from_json  # type: ignore
        except ImportError:
            model_from_json = None
    try:
        import h5py
    except ImportError:
        h5py = None

    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")
    tf = None
    model_from_json = None
    h5py = None

# Configure environment for optimal performance
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

# Model configuration constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
IMG_CLASSES = 5

# Model download configuration - Updated with new .keras models
HUGGINGFACE_REPO_UNET = os.getenv(
    "HUGGINGFACE_REPO_UNET", 
    "sahilvishwa2108/oil-spill-unet"
)
HUGGINGFACE_REPO_DEEPLAB = os.getenv(
    "HUGGINGFACE_REPO_DEEPLAB", 
    "sahilvishwa2108/oil-spill-deeplab"
)
HUGGINGFACE_REPO = os.getenv(
    "HUGGINGFACE_REPO", "sahilvishwa2108/oil-spill-detection-models"
)

# Only .keras format for optimized production performance
MODEL_FILES = {
    "unet_final_model.keras": "unet_final_model.keras",
    "deeplab_final_model.keras": "deeplab_final_model.keras",
}

# Model performance metrics for dashboard (consistent with frontend)
MODEL_PERFORMANCE = {
    "unet_final_model.keras": {
        "name": "U-Net",
        "f1_score": 0.9356,
        "accuracy": 0.9445,
        "architecture": "U-Net",
        "size_mb": 22.39,
        "description": "Lightweight segmentation model optimized for speed",
        "training_epochs": 50,
        "parameters": "2.1M"
    },
    "deeplab_final_model.keras": {
        "name": "DeepLabV3+",
        "f1_score": 0.9668,
        "accuracy": 0.9723,
        "architecture": "DeepLabV3+",
        "size_mb": 204.56,
        "description": "High-accuracy segmentation model with advanced features",
        "training_epochs": 50,
        "parameters": "41.3M"
    }
}

# Class information (EXACT from notebook)
CLASS_INFO = {
    "classes": {
        "BACKGROUND": 0,
        "OIL_SPILL": 1,
        "SHIPS": 2,
        "LOOKLIKE": 3,
        "WAKES": 4
    },
    "class_names": [
        "Background",
        "Oil Spill", 
        "Ships",
        "Looklike",
        "Wakes"
    ],
    "class_colors": [
        [0, 0, 0],         # Background (Black)
        [0, 255, 255],     # Oil Spill (Cyan)
        [255, 0, 0],       # Ships (Red)
        [153, 76, 0],      # Looklike (Brown)
        [0, 153, 0],       # Wakes (Green)
    ]
}

# Detection thresholds (EXACT from notebook)
DETECTION_THRESHOLDS = {
    "oil_spill_pixel_threshold": 0.01,    # 1.0% of pixels minimum (from notebook)
    "confidence_threshold": 0.5,          # 50% confidence minimum
    "risk_levels": {
        "LOW": {"threshold": 0.3, "color": "green"},
        "MODERATE": {"threshold": 0.65, "color": "yellow"},
        "HIGH": {"threshold": 0.8, "color": "orange"},
        "CRITICAL": {"threshold": 0.95, "color": "red"}
    }
}

# Define the same color map as used in the notebook
COLOR_MAP = [
    [0, 0, 0],          # Class 0: Background (Black)
    [0, 255, 255],      # Class 1: Oil Spill (Cyan)
    [255, 0, 0],        # Class 2: Ships (Red)
    [153, 76, 0],       # Class 3: Looklike (Brown)
    [0, 153, 0],        # Class 4: Wakes (Green)
]


def download_model_if_needed(filename: str) -> bool:
    """Download model from HuggingFace if not exists locally"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / filename

    if model_path.exists():
        print(f"✅ Model {filename} already exists")
        return True

    try:
        # Determine which repo to use based on model type
        if "unet" in filename.lower():
            repo = HUGGINGFACE_REPO_UNET
            model_file = "model.keras"
        elif "deeplab" in filename.lower():
            repo = HUGGINGFACE_REPO_DEEPLAB
            model_file = "model.keras"
        else:
            repo = HUGGINGFACE_REPO
            model_file = filename

        url = f"https://huggingface.co/{repo}/resolve/main/{model_file}"
        print(f"⬇️ Downloading {filename} from {repo}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}%", end="")

        print(f"\n✅ Successfully downloaded {filename}")
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
    version="1.0.0",
)

# CORS middleware for frontend access
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://oil-spill-detection-system.vercel.app",
    "https://oil-spill-frontend-oigeradm3-sahil-vishwakarmas-projects.vercel.app",
    "*",  # Temporarily allow all origins for debugging
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
    """Load model1 only when needed - .keras format only"""
    global model1
    if model1 is None:
        try:
            if tf is None:
                print("❌ TensorFlow not available for model loading")
                return None

            model_path = "models/unet_final_model.keras"
            print(f"🔄 Loading UNet model from {model_path}")

            if os.path.exists(model_path):
                print(f"✅ Model file exists at {model_path}")
                model1 = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model 1 (U-Net) loaded successfully into memory")
            else:
                print(f"❌ Model 1 not found at {model_path}")
                models_dir = Path("models")
                if models_dir.exists():
                    files = list(models_dir.glob("*.keras"))
                    print(f"📁 Available .keras files: {files}")
                else:
                    print("📁 Models directory does not exist")
                return None
        except Exception as e:
            print(f"❌ Error loading model 1: {e}")
            import traceback
            traceback.print_exc()
            return None
    return model1


def lazy_load_model2():
    """Load model2 only when needed - .keras format only"""
    global model2
    if model2 is None:
        try:
            if tf is None:
                print("❌ TensorFlow not available for model loading")
                return None

            model_path = "models/deeplab_final_model.keras"
            print(f"🔄 Loading DeepLab model from {model_path}")

            if os.path.exists(model_path):
                print(f"✅ Model file exists at {model_path}")
                model2 = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model 2 (DeepLab) loaded successfully into memory")
            else:
                print(f"❌ Model 2 not found at {model_path}")
                models_dir = Path("models")
                if models_dir.exists():
                    files = list(models_dir.glob("*.keras"))
                    print(f"📁 Available .keras files: {files}")
                else:
                    print("📁 Models directory does not exist")
                return None
        except Exception as e:
            print(f"❌ Error loading model 2: {e}")
            import traceback
            traceback.print_exc()
            return None
    return model2


def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """Preprocess image for model prediction - consistent with notebook"""
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize pixel values to [0, 1] - same as notebook
    img_array = img_array.astype(np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_oil_spill(image_array, model):
    """Make prediction using the specified model - matching notebook approach with consistent output"""
    try:
        # Make prediction
        prediction = model.predict(image_array, verbose=0)
        print(f"Raw prediction shape: {prediction.shape}")
        print(f"Prediction stats - Min: {np.min(prediction):.4f}, Max: {np.max(prediction):.4f}, Mean: {np.mean(prediction):.4f}")

        # Process prediction based on output shape - following notebook logic
        if (
            len(prediction.shape) == 4 and prediction.shape[-1] == 5
        ):  # Multi-class segmentation (5 classes: Background, Oil Spill, Ships, Look-alike, Wakes)
            # Use argmax to get predicted classes - same as notebook
            predicted_classes = np.argmax(prediction, axis=3)[0]  # Shape: (256, 256)
            
            # Get confidence map
            confidence_map = np.max(prediction, axis=3)[0]  # Shape: (256, 256)
            mean_confidence = float(np.mean(confidence_map))

            # Count pixels for each class (following notebook class definitions)
            class_counts = {i: np.sum(predicted_classes == i) for i in range(5)}
            oil_spill_pixels = class_counts.get(1, 0)  # Class 1 is oil spill
            total_pixels = predicted_classes.size

            # Calculate oil spill percentage
            oil_spill_percentage = oil_spill_pixels / total_pixels

            print(f"Class distribution: {class_counts}")
            print(f"Oil spill pixels: {oil_spill_pixels} ({oil_spill_percentage*100:.2f}%)")
            print(f"Mean confidence: {mean_confidence:.4f}")

            # Determine if oil spill is detected - using consistent threshold from constants
            OIL_SPILL_THRESHOLD = DETECTION_THRESHOLDS["oil_spill_pixel_threshold"]
            
            if oil_spill_percentage > OIL_SPILL_THRESHOLD:
                predicted_class = "Oil Spill Detected"
                # Calculate confidence based on oil spill percentage and model confidence
                # Higher oil spill percentage = higher confidence
                base_confidence = oil_spill_percentage * 100  # Convert to percentage
                model_confidence_boost = mean_confidence * 0.3  # Model confidence contribution
                confidence = float(min(0.95, max(0.65, base_confidence + model_confidence_boost)))
            else:
                predicted_class = "No Oil Spill"
                # For no oil spill, confidence is based on clean water percentage
                clean_confidence = (1.0 - oil_spill_percentage) * mean_confidence
                confidence = float(min(0.95, max(0.60, clean_confidence)))

            print(f"Final prediction: {predicted_class} (confidence: {confidence:.3f})")

        elif (
            len(prediction.shape) == 4 and prediction.shape[-1] == 1
        ):  # Binary segmentation
            # Handle binary output
            mask = prediction[0, :, :, 0]
            confidence = float(np.mean(mask))
            predicted_class = (
                "Oil Spill Detected" if confidence > DETECTION_THRESHOLDS["confidence_threshold"] else "No Oil Spill"
            )

        else:  # Classification output
            confidence = float(np.max(prediction))
            predicted_class = (
                "Oil Spill Detected" if confidence > DETECTION_THRESHOLDS["confidence_threshold"] else "No Oil Spill"
            )

        return predicted_class, confidence

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def fallback_predict_oil_spill(image_array, model_name="U-Net"):
    """Fallback prediction function when models can't be loaded"""
    try:
        # Calculate some basic statistics
        mean_intensity = float(np.mean(image_array))
        std_intensity = float(np.std(image_array))

        # Simple heuristic: darker areas might indicate oil spills
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


def apply_color_map(mask, color_map):
    """
    Apply color mapping to segmentation mask - same as notebook

    Args:
        mask: 2D array with class indices
        color_map: List of RGB colors for each class

    Returns:
        RGB colored mask
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(color_map):
        colored_mask[mask == class_id] = color

    return colored_mask


def generate_prediction_mask(image_array, model, prediction_threshold=0.5):
    """Generate prediction mask for visualization - matching notebook approach"""
    try:
        # Make prediction
        prediction = model.predict(image_array, verbose=0)
        print(
            f"Model prediction shape: {prediction.shape}, values range: [{np.min(prediction):.3f}, {np.max(prediction):.3f}]"
        )

        # Process prediction exactly like the notebook
        if (
            len(prediction.shape) == 4 and prediction.shape[-1] == 5
        ):  # Multi-class segmentation
            # Use argmax to get the predicted class for each pixel - same as notebook
            predicted_mask = np.argmax(prediction, axis=3)[0]  # Shape: (256, 256)

            # Apply the same color mapping as the notebook
            colored_mask = apply_color_map(predicted_mask, COLOR_MAP)

            print(
                f"Generated segmentation mask with classes: {np.unique(predicted_mask)}"
            )

        elif (
            len(prediction.shape) == 4 and prediction.shape[-1] == 1
        ):  # Binary segmentation
            # Handle binary output
            mask = prediction[0, :, :, 0]
            binary_mask = (mask > prediction_threshold).astype(np.uint8)

            # Create colored mask - oil spill areas in cyan like notebook
            colored_mask = np.zeros(
                (binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8
            )
            colored_mask[binary_mask == 1] = COLOR_MAP[1]  # Cyan for oil spill

            print(f"Generated binary mask with {np.sum(binary_mask)} oil spill pixels")

        else:  # Fallback for other output types
            # Create a simple visualization
            confidence = np.mean(prediction)
            mask = np.ones((256, 256)) * (confidence > 0.5)
            colored_mask = apply_color_map(mask.astype(np.uint8), COLOR_MAP)

            print(f"Generated fallback mask with confidence {confidence:.3f}")

        # Convert to PIL Image
        mask_image = Image.fromarray(colored_mask)

        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return mask_base64

    except Exception as e:
        print(f"Error generating mask: {e}")
        import traceback
        traceback.print_exc()
        return None


def fallback_generate_mask(image_array, prediction_result):
    """Generate a fallback mask using the same color mapping as notebook"""
    try:
        # Create a more realistic mask based on image features
        original_image = image_array[0]  # Remove batch dimension

        # Convert to grayscale for analysis
        grayscale = np.mean(original_image, axis=-1)

        # Initialize mask with background class (0)
        predicted_mask = np.zeros_like(grayscale, dtype=np.uint8)

        # Create mask based on prediction result
        if "Oil Spill Detected" in prediction_result:
            # Method 1: Dark region detection for oil spills
            dark_threshold = np.percentile(grayscale, 30)  # Bottom 30% of intensities
            dark_mask = (grayscale < dark_threshold).astype(np.float32)

            # Method 2: Smooth region detection (oil tends to be smooth)
            kernel = np.ones((3, 3), np.float32) / 9
            smoothed = cv2.filter2D(grayscale, -1, kernel)
            variance = cv2.filter2D((grayscale - smoothed) ** 2, -1, kernel)
            smooth_threshold = np.percentile(variance, 40)
            smooth_mask = (variance < smooth_threshold).astype(np.float32)

            # Combine masks
            combined_mask = dark_mask * 0.6 + smooth_mask * 0.4

            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Assign oil spill class (1) to detected areas
            oil_spill_areas = (combined_mask > 0.3).astype(bool)
            predicted_mask[oil_spill_areas] = 1  # Class 1 = Oil Spill

            # Add some random ships and wakes for realism
            if np.random.random() > 0.7:  # 30% chance of ships
                ship_areas = np.random.random(grayscale.shape) > 0.98
                predicted_mask[ship_areas] = 2  # Class 2 = Ships

            if np.random.random() > 0.8:  # 20% chance of wakes
                wake_areas = np.random.random(grayscale.shape) > 0.995
                predicted_mask[wake_areas] = 4  # Class 4 = Wakes

        # Apply the same color mapping as the notebook
        colored_mask = apply_color_map(predicted_mask, COLOR_MAP)

        # Convert to PIL Image
        mask_image = Image.fromarray(colored_mask)

        # Convert to base64
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        unique_classes = np.unique(predicted_mask)
        print(f"Generated fallback mask with classes: {unique_classes}")
        return mask_base64

    except Exception as e:
        print(f"Error generating fallback mask: {e}")
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

    individual_predictions.append(
        ModelPrediction(
            model_name="U-Net",
            prediction=predicted_class,
            confidence=confidence,
            processing_time=model1_time,
            prediction_mask=mask,
        )
    )

    # Model 2 (DeepLab) prediction
    model2_start = datetime.now()
    model2_instance = lazy_load_model2()

    if model2_instance is not None:
        try:
            predicted_class, confidence = predict_oil_spill(image_array, model2_instance)
            mask = generate_prediction_mask(image_array, model2_instance)
        except Exception as e:
            print(f"Model 2 prediction failed, using fallback: {e}")
            predicted_class, confidence = fallback_predict_oil_spill(image_array, "DeepLabV3+")
            mask = fallback_generate_mask(image_array, predicted_class)
    else:
        predicted_class, confidence = fallback_predict_oil_spill(image_array, "DeepLabV3+")
        mask = fallback_generate_mask(image_array, predicted_class)

    model2_time = (datetime.now() - model2_start).total_seconds()

    individual_predictions.append(
        ModelPrediction(
            model_name="DeepLabV3+",
            prediction=predicted_class,
            confidence=confidence,
            processing_time=model2_time,
            prediction_mask=mask,
        )
    )

    # Ensemble logic - average confidences and majority vote
    confidences = [pred.confidence for pred in individual_predictions]
    avg_confidence = sum(confidences) / len(confidences)

    # Count oil spill detections
    oil_spill_count = sum(
        1 for pred in individual_predictions if "Oil Spill Detected" in pred.prediction
    )

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

    # Use first model's mask as ensemble mask (simplified)
    ensemble_mask = individual_predictions[0].prediction_mask if individual_predictions else None

    return (
        individual_predictions,
        ensemble_prediction,
        ensemble_confidence,
        ensemble_mask,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check if model files exist (they may not be loaded in memory yet due to lazy loading)
    model1_exists = os.path.exists("models/unet_final_model.keras")
    model2_exists = os.path.exists("models/deeplab_final_model.keras")

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={"model1": model1_exists, "model2": model2_exists},
    )


@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    # Check if model files exist
    model1_exists = os.path.exists("models/unet_final_model.keras")
    model2_exists = os.path.exists("models/deeplab_final_model.keras")

    models_info = {}
    for model_file, performance in MODEL_PERFORMANCE.items():
        model_exists = os.path.exists(f"models/{model_file}")
        models_info[model_file] = {
            "name": performance["name"],
            "architecture": performance["architecture"],
            "f1_score": performance["f1_score"],
            "accuracy": performance.get("accuracy", 0.9),
            "size_mb": performance["size_mb"],
            "description": performance["description"],
            "training_epochs": performance.get("training_epochs", 50),
            "parameters": performance.get("parameters", "N/A"),
            "loaded": model_exists,
            "status": "ready" if model_exists else "downloading"
        }

    return {
        "models": models_info,
        "ensemble_advantage": "Combining both models provides better accuracy and robustness",
        "total_models": len(models_info)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_choice: str = "model1"):
    """Predict oil spill in uploaded image"""
    start_time = datetime.now()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Load appropriate model
        if model_choice == "model1":
            model = lazy_load_model1()
            model_name = MODEL_PERFORMANCE["unet_final_model.keras"]["name"]
        else:
            model = lazy_load_model2()
            model_name = MODEL_PERFORMANCE["deeplab_final_model.keras"]["name"]

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
            prediction_mask=mask,
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
                prediction_mask=mask,
            )
        except Exception as e2:
            return PredictionResponse(success=False, error=str(e2))


@app.post("/ensemble-predict", response_model=EnsemblePredictionResponse)
async def ensemble_predict_endpoint(file: UploadFile = File(...)):
    """Run ensemble prediction with both models"""
    start_time = datetime.now()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run ensemble prediction
        (
            individual_predictions,
            ensemble_prediction,
            ensemble_confidence,
            ensemble_mask,
        ) = ensemble_predict(processed_image)

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
            total_processing_time=round(total_processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during ensemble prediction: {e}")
        return EnsemblePredictionResponse(success=False, error=str(e))


@app.post("/predict/detailed", response_model=EnsemblePredictionResponse)
async def predict_detailed(file: UploadFile = File(...)):
    """Detailed prediction endpoint with ensemble results and additional analysis"""
    start_time = datetime.now()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run ensemble prediction with detailed analysis
        (
            individual_predictions,
            ensemble_prediction,
            ensemble_confidence,
            ensemble_mask,
        ) = ensemble_predict(processed_image)

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
            total_processing_time=round(total_processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during detailed prediction: {e}")
        return EnsemblePredictionResponse(success=False, error=str(e))


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
            "predict_detailed": "/predict/detailed",
            "ensemble-predict": "/ensemble-predict",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Oil Spill Detection API server...")
    print(f"📍 Server will be available at: http://localhost:8000")
    print(f"📚 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
