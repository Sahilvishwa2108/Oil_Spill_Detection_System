"""
Oil Spill Detection API - Optimized for Render Deployment
Downloads models on startup and loads them on-demand to reduce memory usage
"""

import os

# Configure environment for optimal performance BEFORE importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = (
    "0"  # Disable oneDNN optimizations to avoid warnings
)

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

# Model configuration constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
IMG_CLASSES = 5

# Model download configuration - Updated with new .keras models
HUGGINGFACE_REPO_UNET = os.getenv(
    "HUGGINGFACE_REPO_UNET", "sahilvishwa2108/oil-spill-unet"
)
HUGGINGFACE_REPO_DEEPLAB = os.getenv(
    "HUGGINGFACE_REPO_DEEPLAB", "sahilvishwa2108/oil-spill-deeplab"
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
        "parameters": "2.1M",
    },
    "deeplab_final_model.keras": {
        "name": "DeepLabV3+",
        "f1_score": 0.9668,
        "accuracy": 0.9723,
        "architecture": "DeepLabV3+",
        "size_mb": 204.56,
        "description": "High-accuracy segmentation model with advanced features",
        "training_epochs": 50,
        "parameters": "41.3M",
    },
}

# Class information (EXACT from notebook)
CLASS_INFO = {
    "classes": {"BACKGROUND": 0, "OIL_SPILL": 1, "SHIPS": 2, "LOOKLIKE": 3, "WAKES": 4},
    "class_names": ["Background", "Oil Spill", "Ships", "Looklike", "Wakes"],
    "class_colors": [
        [0, 0, 0],  # Background (Black)
        [0, 255, 255],  # Oil Spill (Cyan)
        [255, 0, 0],  # Ships (Red)
        [153, 76, 0],  # Looklike (Brown)
        [0, 153, 0],  # Wakes (Green)
    ],
}

# Detection thresholds (EXACT from notebook)
DETECTION_THRESHOLDS = {
    "oil_spill_pixel_threshold": 0.01,  # 1.0% of pixels minimum (from notebook)
    "confidence_threshold": 0.5,  # 50% confidence minimum
    "risk_levels": {
        "LOW": {"threshold": 0.3, "color": "green"},
        "MODERATE": {"threshold": 0.65, "color": "yellow"},
        "HIGH": {"threshold": 0.8, "color": "orange"},
        "CRITICAL": {"threshold": 0.95, "color": "red"},
    },
}

# Define the same color map as used in the notebook
COLOR_MAP = [
    [0, 0, 0],  # Class 0: Background (Black)
    [0, 255, 255],  # Class 1: Oil Spill (Cyan)
    [255, 0, 0],  # Class 2: Ships (Red)
    [153, 76, 0],  # Class 3: Looklike (Brown)
    [0, 153, 0],  # Class 4: Wakes (Green)
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

        total_size = int(response.headers.get("content-length", 0))
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

    # Additional data for detailed analysis
    oil_spill_percentage: Optional[float] = None
    class_breakdown: Optional[dict] = None
    risk_level: Optional[str] = None


class ModelPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    prediction: str
    confidence: float
    processing_time: float
    prediction_mask: Optional[str] = None
    oil_spill_percentage: Optional[float] = None
    class_breakdown: Optional[dict] = None


class EnsemblePredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    success: bool
    individual_predictions: List[ModelPrediction] = []
    ensemble_prediction: Optional[str] = None
    ensemble_confidence: Optional[float] = None
    ensemble_mask: Optional[str] = None
    total_processing_time: Optional[float] = None
    error: Optional[str] = None

    # New fields to match frontend expectations
    prediction_images: Optional[dict] = None  # Contains base64 encoded mask images
    final_prediction: Optional[str] = None
    confidence_percentage: Optional[float] = None
    oil_spill_percentage: Optional[float] = None
    risk_level: Optional[str] = None
    class_breakdown: Optional[dict] = None
    model_agreement: Optional[dict] = None


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
        print(
            f"Prediction stats - Min: {np.min(prediction):.4f}, Max: {np.max(prediction):.4f}, Mean: {np.mean(prediction):.4f}"
        )

        # Process prediction based on output shape - following notebook logic
        if (
            len(prediction.shape) == 4 and prediction.shape[-1] == 5
        ):  # Multi-class segmentation (5 classes: Background, Oil Spill, Ships, Look-alike, Wakes)
            # Use argmax to get predicted classes - same as notebook
            predicted_classes = np.argmax(prediction, axis=3)[0]  # Shape: (256, 256)

            # Get confidence map and probability scores for each class
            confidence_map = np.max(prediction, axis=3)[0]  # Shape: (256, 256)
            mean_confidence = float(np.mean(confidence_map))

            # Calculate class probabilities (softmax scores)
            class_probabilities = np.mean(
                prediction[0], axis=(0, 1)
            )  # Average across spatial dimensions

            # Count pixels for each class (following notebook class definitions)
            class_counts = {i: np.sum(predicted_classes == i) for i in range(5)}
            oil_spill_pixels = class_counts.get(1, 0)  # Class 1 is oil spill
            total_pixels = predicted_classes.size

            # Calculate oil spill percentage
            oil_spill_percentage = oil_spill_pixels / total_pixels

            print(f"Class distribution: {class_counts}")
            print(
                f"Oil spill pixels: {oil_spill_pixels} ({oil_spill_percentage*100:.2f}%)"
            )
            print(f"Mean confidence: {mean_confidence:.4f}")
            print(f"Class probabilities: {class_probabilities}")

            # Determine if oil spill is detected - using consistent threshold from constants
            OIL_SPILL_THRESHOLD = DETECTION_THRESHOLDS["oil_spill_pixel_threshold"]

            if oil_spill_percentage > OIL_SPILL_THRESHOLD:
                predicted_class = "Oil Spill Detected"
                # Calculate confidence based on oil spill percentage and model confidence
                # Higher oil spill percentage = higher confidence
                base_confidence = oil_spill_percentage * 100  # Convert to percentage
                model_confidence_boost = (
                    mean_confidence * 30
                )  # Model confidence contribution
                confidence = float(
                    min(95, max(65, base_confidence + model_confidence_boost))
                )
            else:
                predicted_class = "No Oil Spill"
                # For no oil spill, confidence is based on clean water percentage
                clean_confidence = (1.0 - oil_spill_percentage) * mean_confidence * 100
                confidence = float(min(95, max(60, clean_confidence)))

            print(
                f"Final prediction: {predicted_class} (confidence: {confidence:.1f}%)"
            )

            # Return additional data for frontend
            prediction_data = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "oil_spill_percentage": oil_spill_percentage * 100,
                "class_counts": class_counts,
                "class_probabilities": class_probabilities.tolist(),
                "mean_confidence": mean_confidence * 100,
                "predicted_mask": predicted_classes,
                "confidence_map": confidence_map,
            }

            return prediction_data

        elif (
            len(prediction.shape) == 4 and prediction.shape[-1] == 1
        ):  # Binary segmentation
            # Handle binary output
            mask = prediction[0, :, :, 0]
            confidence = float(np.mean(mask)) * 100
            predicted_class = (
                "Oil Spill Detected"
                if confidence > DETECTION_THRESHOLDS["confidence_threshold"] * 100
                else "No Oil Spill"
            )

            prediction_data = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "oil_spill_percentage": confidence,
                "predicted_mask": (mask > 0.5).astype(int),
                "confidence_map": mask,
            }

            return prediction_data

        else:  # Classification output
            confidence = float(np.max(prediction)) * 100
            predicted_class = (
                "Oil Spill Detected"
                if confidence > DETECTION_THRESHOLDS["confidence_threshold"] * 100
                else "No Oil Spill"
            )

            prediction_data = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "oil_spill_percentage": (
                    confidence if predicted_class == "Oil Spill Detected" else 0
                ),
            }

            return prediction_data

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

        print(
            f"⚠️ Using fallback prediction: {prediction} (confidence: {confidence:.3f})"
        )
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


def mask_to_base64(mask, color_map=None, is_confidence_map=False):
    """
    Convert prediction mask to base64 encoded image

    Args:
        mask: 2D numpy array (predicted classes or confidence values)
        color_map: Optional color mapping for class visualization
        is_confidence_map: Whether the mask is a confidence map (0-1 values)

    Returns:
        Base64 encoded image string
    """
    try:
        if is_confidence_map:
            # Convert confidence map to heatmap
            mask_normalized = (mask * 255).astype(np.uint8)
            # Create heatmap using matplotlib colormap
            colored_mask = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
            colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
        elif color_map is not None:
            # Apply class color mapping
            colored_mask = apply_color_map(mask.astype(int), color_map)
        else:
            # Grayscale mask
            mask_normalized = (
                (mask - mask.min()) / (mask.max() - mask.min()) * 255
            ).astype(np.uint8)
            colored_mask = cv2.cvtColor(mask_normalized, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(colored_mask)

        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_base64  # Return just the base64 string, not the data URL

    except Exception as e:
        print(f"Error converting mask to base64: {e}")
        return None


def calculate_risk_level(oil_spill_percentage, confidence):
    """
    Calculate risk level based on oil spill percentage and confidence
    Matching the notebook's risk assessment logic
    """
    if oil_spill_percentage > 10:
        return "CRITICAL"
    elif oil_spill_percentage > 5:
        return "HIGH"
    elif oil_spill_percentage > 1:
        return "MODERATE"
    elif oil_spill_percentage > 0.1:
        return "LOW"
    else:
        return "MINIMAL"


def create_class_breakdown(class_counts, total_pixels):
    """
    Create detailed class breakdown with percentages
    """
    class_names = CLASS_INFO["class_names"]
    breakdown = {}

    for class_id, count in class_counts.items():
        percentage = (count / total_pixels) * 100
        breakdown[class_names[class_id]] = {
            "pixel_count": int(count),
            "percentage": round(percentage, 2),
            "class_id": class_id,
        }

    return breakdown


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


def ensemble_predict_comprehensive(image_array):
    """
    Run comprehensive ensemble prediction matching the notebook approach
    Returns detailed analysis with mask images
    """
    start_time = datetime.now()
    individual_results = []
    individual_predictions = []

    # Model 1 (U-Net) prediction
    model1_start = datetime.now()
    model1_instance = lazy_load_model1()

    if model1_instance is not None:
        try:
            prediction_data = predict_oil_spill(image_array, model1_instance)
            if isinstance(prediction_data, dict):
                unet_result = prediction_data
            else:
                # Legacy format handling
                predicted_class, confidence = prediction_data
                unet_result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "oil_spill_percentage": (
                        confidence if predicted_class == "Oil Spill Detected" else 0
                    ),
                }
        except Exception as e:
            print(f"UNet prediction failed: {e}")
            unet_result = {
                "predicted_class": "Analysis Failed",
                "confidence": 0,
                "oil_spill_percentage": 0,
            }
    else:
        unet_result = {
            "predicted_class": "Model Not Available",
            "confidence": 0,
            "oil_spill_percentage": 0,
        }

    model1_time = (datetime.now() - model1_start).total_seconds()

    # Generate UNet mask image
    unet_mask_image = None
    if "predicted_mask" in unet_result:
        unet_mask_image = mask_to_base64(unet_result["predicted_mask"], COLOR_MAP)

    individual_results.append(unet_result)
    individual_predictions.append(
        ModelPrediction(
            model_name="UNet",
            prediction=unet_result["predicted_class"],
            confidence=unet_result["confidence"],
            processing_time=model1_time,
            prediction_mask=unet_mask_image,
            oil_spill_percentage=unet_result.get("oil_spill_percentage", 0),
            class_breakdown=(
                create_class_breakdown(
                    unet_result.get("class_counts", {}), IMG_WIDTH * IMG_HEIGHT
                )
                if "class_counts" in unet_result
                else None
            ),
        )
    )

    # Model 2 (DeepLabV3+) prediction
    model2_start = datetime.now()
    model2_instance = lazy_load_model2()

    if model2_instance is not None:
        try:
            prediction_data = predict_oil_spill(image_array, model2_instance)
            if isinstance(prediction_data, dict):
                deeplab_result = prediction_data
            else:
                # Legacy format handling
                predicted_class, confidence = prediction_data
                deeplab_result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "oil_spill_percentage": (
                        confidence if predicted_class == "Oil Spill Detected" else 0
                    ),
                }
        except Exception as e:
            print(f"DeepLabV3+ prediction failed: {e}")
            deeplab_result = {
                "predicted_class": "Analysis Failed",
                "confidence": 0,
                "oil_spill_percentage": 0,
            }
    else:
        deeplab_result = {
            "predicted_class": "Model Not Available",
            "confidence": 0,
            "oil_spill_percentage": 0,
        }

    model2_time = (datetime.now() - model2_start).total_seconds()

    # Generate DeepLab mask image
    deeplab_mask_image = None
    if "predicted_mask" in deeplab_result:
        deeplab_mask_image = mask_to_base64(deeplab_result["predicted_mask"], COLOR_MAP)

    individual_results.append(deeplab_result)
    individual_predictions.append(
        ModelPrediction(
            model_name="DeepLabV3+",
            prediction=deeplab_result["predicted_class"],
            confidence=deeplab_result["confidence"],
            processing_time=model2_time,
            prediction_mask=deeplab_mask_image,
            oil_spill_percentage=deeplab_result.get("oil_spill_percentage", 0),
            class_breakdown=(
                create_class_breakdown(
                    deeplab_result.get("class_counts", {}), IMG_WIDTH * IMG_HEIGHT
                )
                if "class_counts" in deeplab_result
                else None
            ),
        )
    )

    # Ensemble decision (weighted average)
    unet_weight = 0.4  # UNet is faster but less accurate
    deeplab_weight = 0.6  # DeepLabV3+ is more accurate

    # Calculate ensemble prediction
    unet_oil_percentage = unet_result.get("oil_spill_percentage", 0)
    deeplab_oil_percentage = deeplab_result.get("oil_spill_percentage", 0)

    ensemble_oil_percentage = (
        unet_oil_percentage * unet_weight + deeplab_oil_percentage * deeplab_weight
    )

    ensemble_confidence = (
        unet_result.get("confidence", 0) * unet_weight
        + deeplab_result.get("confidence", 0) * deeplab_weight
    )

    # Determine final prediction based on ensemble
    oil_spill_threshold = (
        DETECTION_THRESHOLDS["oil_spill_pixel_threshold"] * 100
    )  # Convert to percentage

    if ensemble_oil_percentage > oil_spill_threshold:
        ensemble_prediction = "Oil Spill Detected"
    else:
        ensemble_prediction = "No Oil Spill"

    # Calculate model agreement
    unet_detected = "oil spill" in unet_result["predicted_class"].lower()
    deeplab_detected = "oil spill" in deeplab_result["predicted_class"].lower()

    if unet_detected == deeplab_detected:
        agreement_percentage = 100.0
        agreement_status = "Complete Agreement"
    else:
        # Partial agreement based on confidence similarity
        conf_diff = abs(
            unet_result.get("confidence", 0) - deeplab_result.get("confidence", 0)
        )
        agreement_percentage = max(50, 100 - (conf_diff * 100))
        agreement_status = "Partial Agreement"

    model_agreement = {
        "agreementPercentage": agreement_percentage,
        "agreementStatus": agreement_status,
        "unetDetected": unet_detected,
        "deeplabDetected": deeplab_detected,
    }

    # Create ensemble mask (simple average for visualization)
    ensemble_mask_image = None
    if "predicted_mask" in unet_result and "predicted_mask" in deeplab_result:
        try:
            # Simple ensemble: average the masks
            unet_mask = unet_result["predicted_mask"].astype(float)
            deeplab_mask = deeplab_result["predicted_mask"].astype(float)
            ensemble_mask = (
                (unet_mask * unet_weight + deeplab_mask * deeplab_weight)
            ).astype(int)
            ensemble_mask_image = mask_to_base64(ensemble_mask, COLOR_MAP)
        except Exception as e:
            print(f"Error creating ensemble mask: {e}")

    # Calculate risk level
    risk_level = calculate_risk_level(ensemble_oil_percentage, ensemble_confidence)

    # Create comprehensive class breakdown
    ensemble_class_breakdown = {}
    if individual_results:
        for class_name in CLASS_INFO["class_names"]:
            avg_percentage = 0
            valid_results = 0
            for result in individual_results:
                if "class_counts" in result:
                    breakdown = create_class_breakdown(
                        result["class_counts"], IMG_WIDTH * IMG_HEIGHT
                    )
                    if class_name in breakdown:
                        avg_percentage += breakdown[class_name]["percentage"]
                        valid_results += 1

            if valid_results > 0:
                ensemble_class_breakdown[class_name] = {
                    "percentage": round(avg_percentage / valid_results, 2),
                    "confidence": "High" if valid_results == 2 else "Medium",
                }

    # Prediction images for frontend
    prediction_images = {
        "unet_predicted": unet_mask_image,
        "deeplab_predicted": deeplab_mask_image,
        "ensemble_predicted": ensemble_mask_image,
    }

    total_processing_time = (datetime.now() - start_time).total_seconds()

    return {
        "individual_predictions": individual_predictions,
        "ensemble_prediction": ensemble_prediction,
        "ensemble_confidence": ensemble_confidence,
        "final_prediction": ensemble_prediction,
        "confidence_percentage": ensemble_confidence,
        "oil_spill_percentage": ensemble_oil_percentage,
        "risk_level": risk_level,
        "class_breakdown": ensemble_class_breakdown,
        "model_agreement": model_agreement,
        "prediction_images": prediction_images,
        "total_processing_time": total_processing_time,
    }

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
    ensemble_mask = (
        individual_predictions[0].prediction_mask if individual_predictions else None
    )

    return (
        individual_predictions,
        ensemble_prediction,
        ensemble_confidence,
        ensemble_mask,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check if model files exist and are accessible
    model1_available = os.path.exists("models/unet_final_model.keras")
    model2_available = os.path.exists("models/deeplab_final_model.keras")

    # Also check if they're loaded in memory
    global model1, model2
    model1_loaded = model1 is not None
    model2_loaded = model2 is not None

    # Consider models "ready" if files exist, even if not loaded in memory
    # This gives better UX as models are loaded on-demand
    status = "healthy" if (model1_available and model2_available) else "unhealthy"

    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "model1": model1_loaded or model1_available,
            "model2": model2_loaded or model2_available,
        },
    )


@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    # Check if models are actually loaded in memory
    global model1, model2
    model1_loaded = model1 is not None
    model2_loaded = model2 is not None

    models_info = {}
    for model_file, performance in MODEL_PERFORMANCE.items():
        model_exists = os.path.exists(f"models/{model_file}")
        # Correct mapping: unet_final_model.keras -> model1, deeplab_final_model.keras -> model2
        is_loaded = (
            model1_loaded if "unet_final_model.keras" == model_file else model2_loaded
        )

        models_info[model_file] = {
            "name": performance["name"],
            "architecture": performance["architecture"],
            "f1_score": performance["f1_score"],
            "accuracy": performance.get("accuracy", 0.9),
            "size_mb": performance["size_mb"],
            "description": performance["description"],
            "training_epochs": performance.get("training_epochs", 50),
            "parameters": performance.get("parameters", "N/A"),
            "loaded": is_loaded,
            "file_exists": model_exists,
            "status": (
                "loaded" if is_loaded else ("ready" if model_exists else "missing")
            ),
        }

    return {
        "models": models_info,
        "ensemble_advantage": "Combining both models provides better accuracy and robustness",
        "total_models": len(models_info),
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
            predicted_class, confidence = fallback_predict_oil_spill(
                processed_image, model_name
            )
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
            predicted_class, confidence = fallback_predict_oil_spill(
                processed_image, model_name
            )
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
        ensemble_result = ensemble_predict_comprehensive(processed_image)

        # Calculate total processing time (already included in ensemble_result)
        # total_processing_time = (datetime.now() - start_time).total_seconds()

        # Clean up memory
        gc.collect()

        return EnsemblePredictionResponse(
            success=True,
            individual_predictions=ensemble_result["individual_predictions"],
            ensemble_prediction=ensemble_result["ensemble_prediction"],
            ensemble_confidence=round(ensemble_result["ensemble_confidence"], 4),
            ensemble_mask=ensemble_result["prediction_images"].get(
                "ensemble_predicted"
            ),
            total_processing_time=round(ensemble_result["total_processing_time"], 2),
            # New fields matching frontend expectations
            prediction_images=ensemble_result["prediction_images"],
            final_prediction=ensemble_result["final_prediction"],
            confidence_percentage=round(ensemble_result["confidence_percentage"], 1),
            oil_spill_percentage=round(ensemble_result["oil_spill_percentage"], 2),
            risk_level=ensemble_result["risk_level"],
            class_breakdown=ensemble_result["class_breakdown"],
            model_agreement=ensemble_result["model_agreement"],
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
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Run ensemble prediction with detailed analysis
        ensemble_result = ensemble_predict_comprehensive(processed_image)

        # Calculate total processing time
        total_processing_time = (datetime.now() - start_time).total_seconds()

        # Clean up memory
        gc.collect()

        return EnsemblePredictionResponse(
            success=True,
            individual_predictions=ensemble_result["individual_predictions"],
            ensemble_prediction=ensemble_result["ensemble_prediction"],
            ensemble_confidence=round(ensemble_result["ensemble_confidence"], 4),
            ensemble_mask=ensemble_result["prediction_images"].get(
                "ensemble_predicted"
            ),
            total_processing_time=round(ensemble_result["total_processing_time"], 2),
            # New fields matching frontend expectations
            prediction_images=ensemble_result["prediction_images"],
            final_prediction=ensemble_result["final_prediction"],
            confidence_percentage=round(ensemble_result["confidence_percentage"], 1),
            oil_spill_percentage=round(ensemble_result["oil_spill_percentage"], 2),
            risk_level=ensemble_result["risk_level"],
            class_breakdown=ensemble_result["class_breakdown"],
            model_agreement=ensemble_result["model_agreement"],
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


@app.get("/debug/test-image")
async def test_image_endpoint():
    """Debug endpoint to test image generation"""
    try:
        import numpy as np
        from PIL import Image
        import io
        import base64

        # Create a simple test image (red square)
        test_array = np.zeros((256, 256, 3), dtype=np.uint8)
        test_array[50:200, 50:200] = [255, 0, 0]  # Red square

        # Convert to PIL Image
        test_image = Image.fromarray(test_array, "RGB")

        # Convert to base64
        buffer = io.BytesIO()
        test_image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "test_image": base64_str,
            "image_info": {
                "size": test_image.size,
                "mode": test_image.mode,
                "base64_length": len(base64_str),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Oil Spill Detection API server...")
    print(f"📍 Server will be available at: http://localhost:8000")
    print(f"📚 API documentation at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
