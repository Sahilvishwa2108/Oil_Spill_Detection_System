export interface PredictionResult {
  success: boolean;
  prediction?: string;
  confidence?: number;
  processing_time?: number;
  model_used?: string;
  error?: string;
}

export interface ModelInfo {
  name: string;
  description: string;
  loaded: boolean;
}

export interface ModelsResponse {
  models: {
    model1: ModelInfo;
    model2: ModelInfo;
  };
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
