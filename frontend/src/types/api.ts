// API Types and Interfaces
export interface PredictionResponse {
  filename: string;
  prediction: "NORMAL" | "PNEUMONIA";
  confidence: number;
  probability_normal: number;
  probability_pneumonia: number;
  processing_time_seconds: number;
  timestamp: string;
  medical_advice: string;
  model_version: string;
}

export interface UploadResponse {
  message: string;
  uploaded_files: UploadedFile[];
  total_files: number;
  timestamp: string;
  ready_for_training: boolean;
}

export interface UploadedFile {
  filename: string;
  label: "NORMAL" | "PNEUMONIA";
  saved_path: string;
  size_bytes: number;
}

export interface RetrainResponse {
  message: string;
  training_id: string;
  status: TrainingStatus;
  estimated_completion_minutes: number;
  parameters: TrainingParameters;
  timestamp: string;
}

export interface TrainingParameters {
  epochs: number;
  learning_rate: number;
  use_uploaded_data: boolean;
}

export interface StatusResponse {
  api_status: string;
  model_loaded: boolean;
  model_version: string;
  uptime_seconds: number;
  last_prediction_time: string | null;
  total_predictions: number;
  memory_usage_mb: number;
  gpu_available: boolean;
  training_status: TrainingStatus;
  model_accuracy: number | null;
  model_path: string;
  timestamp: string;
}

export enum TrainingStatus {
  IDLE = "IDLE",
  STARTING = "STARTING",
  TRAINING = "TRAINING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
}

// Chart Data Types
export interface AccuracyData {
  epoch: number;
  accuracy: number;
  val_accuracy: number;
  loss: number;
  val_loss: number;
}

export interface ClassDistribution {
  label: string;
  count: number;
  percentage: number;
}

export interface UploadLog {
  timestamp: string;
  filename: string;
  label: "NORMAL" | "PNEUMONIA";
  size_bytes: number;
}

export interface TrainingLog {
  training_id: string;
  start_time: string;
  end_time?: string;
  status: TrainingStatus;
  epochs: number;
  final_accuracy?: number;
  parameters: TrainingParameters;
}
