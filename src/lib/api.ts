// API Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Custom API Error class
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public response?: Response
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// API Client with enhanced error handling
export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async handleResponse(response: Response) {
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.error || errorMessage;
      } catch {
        // If response is not JSON, use status text
      }
      
      throw new ApiError(response.status, errorMessage, response);
    }
      try {
      return await response.json();
    } catch {
      throw new ApiError(500, 'Invalid JSON response from server');
    }
  }

  private async makeRequest(url: string, options?: RequestInit) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...options?.headers,
        },
      });
      
      return await this.handleResponse(response);
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      
      // Network errors, timeout, etc.
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        throw new ApiError(0, 'Unable to connect to the server. Please check your internet connection and try again.');
      }
      
      throw new ApiError(500, error instanceof Error ? error.message : 'An unexpected error occurred');
    }
  }

  async healthCheck() {
    return this.makeRequest(`${this.baseUrl}/health`);
  }

  async getModelsInfo() {
    return this.makeRequest(`${this.baseUrl}/models/info`);
  }
  async predictOilSpill(file: File, modelChoice: string = 'model1') {
    if (!file) {
      throw new ApiError(400, 'No file provided');
    }
    
    if (!file.type.startsWith('image/')) {
      throw new ApiError(400, 'File must be an image');
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      throw new ApiError(400, 'File size must be less than 10MB');
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_choice', modelChoice);

    return this.makeRequest(`${this.baseUrl}/predict`, {
      method: 'POST',
      body: formData,
    });
  }

  async ensemblePredict(file: File) {
    if (!file) {
      throw new ApiError(400, 'No file provided');
    }
    
    if (!file.type.startsWith('image/')) {
      throw new ApiError(400, 'File must be an image');
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      throw new ApiError(400, 'File size must be less than 10MB');
    }

    const formData = new FormData();
    formData.append('file', file);

    return this.makeRequest(`${this.baseUrl}/ensemble-predict`, {
      method: 'POST',
      body: formData,
    });
  }

  async batchPredict(files: File[], modelChoice: string = 'model1') {
    if (!files || files.length === 0) {
      throw new ApiError(400, 'No files provided');
    }
    
    if (files.length > 5) {
      throw new ApiError(400, 'Maximum 5 files allowed for batch prediction');
    }

    const formData = new FormData();
    files.forEach(file => {
      if (!file.type.startsWith('image/')) {
        throw new ApiError(400, `File ${file.name} is not an image`);
      }
      formData.append('files', file);
    });
    formData.append('model_choice', modelChoice);

    return this.makeRequest(`${this.baseUrl}/batch-predict`, {
      method: 'POST',
      body: formData,
    });
  }

  async detailedEnsemblePredict(file: File) {
    if (!file) {
      throw new ApiError(400, 'No file provided');
    }
    
    if (!file.type.startsWith('image/')) {
      throw new ApiError(400, 'File must be an image');
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      throw new ApiError(400, 'File size must be less than 10MB');
    }

    const formData = new FormData();
    formData.append('file', file);

    return this.makeRequest(`${this.baseUrl}/predict/detailed`, {
      method: 'POST',
      body: formData,
    });
  }
}

export const apiClient = new ApiClient();
