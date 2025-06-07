export interface PredictionResult {
  success: boolean;
  prediction_mask?: string;
  confidence_score?: number;
  processing_time?: number;
  selected_model?: string;
  error?: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  input_shape: number[];
  output_shape: number[];
  parameters: number;
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
