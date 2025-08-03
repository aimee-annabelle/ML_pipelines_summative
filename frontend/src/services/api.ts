import axios from "axios";
import {
  PredictionResponse,
  UploadResponse,
  RetrainResponse,
  StatusResponse,
  TrainingParameters,
} from "../types/api";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for file uploads and training
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(
      `Making ${config.method?.toUpperCase()} request to: ${config.url}`
    );
    return config;
  },
  (error) => {
    console.error("Request error:", error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Predict pneumonia from X-ray image
  async predictPneumonia(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post<PredictionResponse>("/predict", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  // Upload training images
  async uploadTrainingImages(
    files: File[],
    labels: string[]
  ): Promise<UploadResponse> {
    const formData = new FormData();

    files.forEach((file) => {
      formData.append("files", file);
    });

    formData.append("labels", labels.join(","));

    const response = await api.post<UploadResponse>("/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  // Trigger model retraining
  async retrainModel(parameters: TrainingParameters): Promise<RetrainResponse> {
    const formData = new FormData();
    formData.append(
      "use_uploaded_data",
      parameters.use_uploaded_data.toString()
    );
    formData.append("epochs", parameters.epochs.toString());
    formData.append("learning_rate", parameters.learning_rate.toString());

    const response = await api.post<RetrainResponse>("/retrain", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  // Get model and API status
  async getStatus(): Promise<StatusResponse> {
    const response = await api.get<StatusResponse>("/status");
    return response.data;
  },

  // Health check
  async healthCheck(): Promise<{ status: string }> {
    try {
      await api.get("/");
      return { status: "healthy" };
    } catch (error) {
      return { status: "unhealthy" };
    }
  },
};

export default apiService;
