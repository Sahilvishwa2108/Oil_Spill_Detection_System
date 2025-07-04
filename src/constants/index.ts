// Project Constants - Single source of truth for all project-wide constants
// This file ensures consistency across the entire application

// === PROJECT METADATA ===
export const PROJECT_INFO = {
  name: "Oil Spill Detection System",
  version: "1.0.0",
  description: "AI-powered oil spill detection using deep learning models with real-time satellite image analysis",
  author: "Sahil Vishwakarma",
  repository: "https://github.com/sahilvishwa2108/oil-spill-detection"
} as const;

// === MODEL CONFIGURATION ===
export const MODEL_CONFIG = {
  // Image processing constants
  IMG_WIDTH: 256,
  IMG_HEIGHT: 256,
  IMG_CHANNELS: 3,
  IMG_CLASSES: 5,
  
  // Model names (standardized - matching backend)
  MODEL_NAMES: {
    UNET: "U-Net",
    DEEPLAB: "DeepLabV3+"
  } as const,
  
  // Model files
  MODEL_FILES: {
    UNET: "unet_final_model.keras",
    DEEPLAB: "deeplab_final_model.keras"
  },
  
  // Performance metrics (from actual training results)
  MODEL_PERFORMANCE: {
    UNET: {
      name: "U-Net",
      f1_score: 0.9356,
      accuracy: 0.9445,
      architecture: "U-Net",
      size_mb: 22.39,
      description: "Lightweight segmentation model optimized for speed",
      training_epochs: 50,
      parameters: "2.1M"
    },
    DEEPLAB: {
      name: "DeepLabV3+",
      f1_score: 0.9668,
      accuracy: 0.9723,
      architecture: "DeepLabV3+",
      size_mb: 204.56,
      description: "High-accuracy segmentation model with advanced features",
      training_epochs: 50,
      parameters: "41.3M"
    }
  }
} as const;

// === CLASS DEFINITIONS ===
export const CLASS_INFO = {
  // Class indices (matching the notebook training)
  CLASSES: {
    BACKGROUND: 0,
    OIL_SPILL: 1,
    SHIPS: 2,
    LOOKLIKE: 3,
    WAKES: 4
  },
  
  // Class names (EXACT from notebook)
  CLASS_NAMES: [
    "Background",
    "Oil Spill",
    "Ships",
    "Looklike",
    "Wakes"
  ] as const,
  
  // Class colors (RGB values matching notebook)
  CLASS_COLORS: [
    [0, 0, 0],         // Background (Black)
    [0, 255, 255],     // Oil Spill (Cyan)
    [255, 0, 0],       // Ships (Red)
    [153, 76, 0],      // Look-alike (Brown)
    [0, 153, 0],       // Wakes (Green)
  ] as const,
  
  // Class descriptions (EXACT from notebook)
  CLASS_DESCRIPTIONS: {
    BACKGROUND: "Clean water or background areas",
    OIL_SPILL: "Oil spill contamination areas",
    SHIPS: "Ship vessels and structures",
    LOOKLIKE: "False positive areas that resemble oil spills",
    WAKES: "Ship wake patterns on water surface"
  } as const,
  
  // Class icons for UI
  CLASS_ICONS: {
    BACKGROUND: "🌊",
    OIL_SPILL: "🛢️",
    SHIPS: "🚢",
    LOOKLIKE: "⚠️",
    WAKES: "〰️"
  } as const
} as const;

// === DETECTION THRESHOLDS ===
export const DETECTION_THRESHOLDS = {
  // Oil spill detection thresholds (matching notebook - 1.0% threshold)
  OIL_SPILL_PIXEL_THRESHOLD: 0.01,  // 1.0% of pixels minimum (from notebook)
  CONFIDENCE_THRESHOLD: 0.5,         // 50% confidence minimum
  
  // Risk levels (matching backend)
  RISK_LEVELS: {
    LOW: { threshold: 0.3, color: "green" },
    MODERATE: { threshold: 0.65, color: "yellow" },
    HIGH: { threshold: 0.8, color: "orange" },
    CRITICAL: { threshold: 0.95, color: "red" }
  }
} as const;

// === UI CONSTANTS ===
export const UI_CONFIG = {
  // Image categories for test gallery
  CATEGORIES: ['all', 'satellite', 'coastal', 'offshore', 'complex'] as const,
  
  // Difficulty levels
  DIFFICULTIES: ['all', 'easy', 'medium', 'hard'] as const,
  
  // Category colors
  DIFFICULTY_COLORS: {
    easy: "bg-green-100 text-green-800 border-green-200",
    medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
    hard: "bg-red-100 text-red-800 border-red-200"
  } as const,
  
  // Category icons
  CATEGORY_ICONS: {
    satellite: "🛰️",
    coastal: "🏖️",
    offshore: "🌊",
    complex: "🌀",
    "test-data": "📊"
  } as const,
  
  // Animation timings
  ANIMATION_TIMING: {
    FAST: 0.3,
    MEDIUM: 0.5,
    SLOW: 0.8
  } as const
} as const;

// === API CONSTANTS ===
export const API_CONFIG = {
  // Base URLs
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  
  // Endpoints
  ENDPOINTS: {
    HEALTH: '/health',
    MODELS_INFO: '/models/info',
    PREDICT: '/predict',
    ENSEMBLE_PREDICT: '/ensemble-predict',
    BATCH_PREDICT: '/batch-predict'
  } as const,
  
  // File upload limits
  FILE_LIMITS: {
    MAX_SIZE_MB: 10,
    MAX_FILES: 5,
    ACCEPTED_TYPES: ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff']
  } as const
} as const;

// === ENSEMBLE PREDICTION CONSTANTS ===
export const ENSEMBLE_CONFIG = {
  // Ensemble decision making
  AGREEMENT_BOOST: 0.1,      // Confidence boost when models agree
  DISAGREEMENT_PENALTY: 0.1, // Confidence penalty when models disagree
  
  // Ensemble weights (can be adjusted based on model performance)
  MODEL_WEIGHTS: {
    UNET: 0.4,      // Lower weight due to lower F1 score
    DEEPLAB: 0.6    // Higher weight due to higher F1 score
  } as const
} as const;

// === TEST DATA CONSTANTS ===
export const TEST_DATA = {
  TOTAL_IMAGES: 110,
  RANDOM_SAMPLE_SIZE: 20,
  CLOUDINARY_BASE_URL: "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637884/oil-spill-test-images"
} as const;

// Type exports for TypeScript
export type ModelName = typeof MODEL_CONFIG.MODEL_NAMES[keyof typeof MODEL_CONFIG.MODEL_NAMES];
export type ClassName = typeof CLASS_INFO.CLASS_NAMES[number];
export type Category = typeof UI_CONFIG.CATEGORIES[number];
export type Difficulty = typeof UI_CONFIG.DIFFICULTIES[number];
export type RiskLevel = keyof typeof DETECTION_THRESHOLDS.RISK_LEVELS;
