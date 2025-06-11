// API Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// API Client
export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async getModelsInfo() {
    const response = await fetch(`${this.baseUrl}/models/info`);
    return response.json();
  }

  async predictOilSpill(file: File, modelChoice: string = 'model1') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_choice', modelChoice);

    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }

  async batchPredict(files: File[], modelChoice: string = 'model1') {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('model_choice', modelChoice);

    const response = await fetch(`${this.baseUrl}/batch-predict`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }
}

export const apiClient = new ApiClient();
