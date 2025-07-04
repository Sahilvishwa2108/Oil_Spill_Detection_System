import { ModelName, ClassName } from "@/constants";

export interface PredictionResult {
  success: boolean;
  prediction?: string;
  confidence?: number;
  processing_time?: number;
  model_used?: ModelName;
  error?: string;
  prediction_mask?: string; // Base64 encoded image
}

export interface ModelPrediction {
  model_name: ModelName;
  prediction: string;
  confidence: number;
  processing_time: number;
  prediction_mask?: string;
}

export interface EnsemblePredictionResult {
  success: boolean;
  individual_predictions: ModelPrediction[];
  ensemble_prediction: string;
  ensemble_confidence: number;
  ensemble_mask?: string;
  total_processing_time: number;
  error?: string;
}

export interface ModelInfo {
  name: ModelName;
  description: string;
  loaded: boolean;
  architecture?: string;
  f1_score?: number;
  accuracy?: number;
  size_mb?: number;
  parameters?: string;
  training_epochs?: number;
  status?: string;
}

export interface ModelsResponse {
  models: Record<string, ModelInfo>;
  ensemble_advantage?: string;
  total_models?: number;
}

export interface HealthStatus {
  status: string;
  models_loaded: {
    model1: boolean;
    model2: boolean;
  };
  timestamp: string;
}

export interface BatchResult {
  filename: string;
  result: PredictionResult;
}

export interface BatchPredictionResponse {
  batch_results: BatchResult[];
}
