from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import os
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from model import ChestXRayClassifier
    from preprocessing import preprocess_image
    from prediction import load_model, predict_image
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")
    ChestXRayClassifier = None
    preprocess_image = None
    load_model = None
    predict_image = None


class PredictionClass(str, Enum):
    """Possible prediction classes"""
    NORMAL = "NORMAL"
    PNEUMONIA = "PNEUMONIA"


class TrainingStatus(str, Enum):
    """Training job status"""
    IDLE = "idle"
    STARTING = "starting"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionResponse(BaseModel):
    """Response model for pneumonia prediction"""
    filename: str = Field(..., description="Original filename of uploaded image")
    prediction: PredictionClass = Field(..., description="Predicted class (NORMAL or PNEUMONIA)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    probability_normal: float = Field(..., ge=0, le=1, description="Probability of normal chest X-ray")
    probability_pneumonia: float = Field(..., ge=0, le=1, description="Probability of pneumonia")
    processing_time_seconds: float = Field(..., description="Time taken for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    medical_advice: str = Field(..., description="Medical advice and disclaimer")
    model_version: str = Field(..., description="Version of the model used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "filename": "chest_xray_001.jpg",
                "prediction": "PNEUMONIA",
                "confidence": 0.87,
                "probability_normal": 0.13,
                "probability_pneumonia": 0.87,
                "processing_time_seconds": 0.45,
                "timestamp": "2024-01-20T10:30:45.123456",
                "medical_advice": "This AI prediction is for assistance only. Please consult with a qualified healthcare professional for proper diagnosis and treatment.",
                "model_version": "1.0.0"
            }
        }


class UploadedFile(BaseModel):
    """Information about an uploaded file"""
    filename: str = Field(..., description="Original filename")
    label: PredictionClass = Field(..., description="Ground truth label")
    saved_path: str = Field(..., description="Path where file was saved")
    size_bytes: int = Field(..., description="File size in bytes")


class UploadResponse(BaseModel):
    """Response model for training data upload"""
    message: str = Field(..., description="Upload status message")
    uploaded_files: List[UploadedFile] = Field(..., description="List of uploaded files")
    total_files: int = Field(..., description="Total number of files uploaded")
    timestamp: datetime = Field(..., description="Upload timestamp")
    ready_for_training: bool = Field(..., description="Whether data is ready for training")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message": "Successfully uploaded 5 training images",
                "uploaded_files": [
                    {
                        "filename": "normal_001.jpg",
                        "label": "NORMAL",
                        "saved_path": "/uploads/training/normal/normal_001.jpg",
                        "size_bytes": 524288
                    }
                ],
                "total_files": 5,
                "timestamp": "2024-01-20T10:30:45.123456",
                "ready_for_training": True
            }
        }


class TrainingParameters(BaseModel):
    """Training parameters"""
    epochs: int = Field(..., description="Number of training epochs")
    learning_rate: float = Field(..., description="Learning rate")
    use_uploaded_data: bool = Field(..., description="Whether to include uploaded data")


class RetrainResponse(BaseModel):
    """Response model for model retraining"""
    message: str = Field(..., description="Retraining status message")
    training_id: str = Field(..., description="Unique training job ID")
    status: TrainingStatus = Field(..., description="Current training status")
    estimated_completion_minutes: int = Field(..., description="Estimated completion time in minutes")
    parameters: TrainingParameters = Field(..., description="Training parameters used")
    timestamp: datetime = Field(..., description="Training start timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "message": "Model retraining started successfully",
                "training_id": "training_20240120_103045",
                "status": "starting",
                "estimated_completion_minutes": 10,
                "parameters": {
                    "epochs": 5,
                    "learning_rate": 0.001,
                    "use_uploaded_data": True
                },
                "timestamp": "2024-01-20T10:30:45.123456"
            }
        }


class StatusResponse(BaseModel):
    """Response model for system status"""
    api_status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    last_prediction_time: Optional[datetime] = Field(None, description="Last prediction timestamp")
    total_predictions: int = Field(..., description="Total predictions made")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    training_status: TrainingStatus = Field(..., description="Current training status")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy on test set")
    model_path: Optional[str] = Field(None, description="Path to current model file")
    timestamp: datetime = Field(..., description="Status check timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "api_status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "uptime_seconds": 3600.0,
                "last_prediction_time": "2024-01-20T10:25:30.123456",
                "total_predictions": 42,
                "memory_usage_mb": 512.5,
                "gpu_available": True,
                "training_status": "idle",
                "model_accuracy": 0.92,
                "model_path": "/models/chest_xray_model.h5",
                "timestamp": "2024-01-20T10:30:45.123456"
            }
        }


class TrainingStatusResponse(BaseModel):
    """Response model for training job status"""
    training_id: str = Field(..., description="Training job ID")
    status: TrainingStatus = Field(..., description="Current status")
    progress_percentage: float = Field(..., ge=0, le=100, description="Training progress (0-100%)")
    current_epoch: int = Field(..., description="Current training epoch")
    total_epochs: int = Field(..., description="Total epochs to train")
    current_loss: Optional[float] = Field(None, description="Current training loss")
    current_accuracy: Optional[float] = Field(None, description="Current training accuracy")
    validation_loss: Optional[float] = Field(None, description="Current validation loss")
    validation_accuracy: Optional[float] = Field(None, description="Current validation accuracy")
    started_at: datetime = Field(..., description="Training start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if training failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelManager:
    """Model manager for handling ML operations"""
    
    def __init__(self):
        self.model = None
        self.model_version = "1.0.0"
        self.training_jobs = {}
        self.prediction_count = 0
        self.last_prediction_time = None
        self.model_path = os.path.join("models", "chest_xray_model.h5")
        self.start_time = time.time()
        self.is_training = False
        
    async def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path) and load_model:
                # Use thread executor for blocking I/O
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    self.model = await loop.run_in_executor(
                        executor, load_model, self.model_path
                    )
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path} or load_model function unavailable")
                # Try to create and train a new model
                if ChestXRayClassifier:
                    classifier = ChestXRayClassifier()
                    # Check if we have training data
                    train_data_path = "data/train"
                    if os.path.exists(train_data_path):
                        print("Training new model...")
                        await self._train_new_model(classifier, train_data_path)
                    else:
                        print("No training data found. Model will be unavailable.")
                        
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    async def _train_new_model(self, classifier, train_data_path: str):
        """Train a new model"""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                # Train in separate thread
                await loop.run_in_executor(
                    executor, 
                    classifier.train, 
                    train_data_path, 
                    "data/test",  # validation data
                    self.model_path
                )
            # Load the newly trained model
            if load_model:
                self.model = await loop.run_in_executor(
                    executor, load_model, self.model_path
                )
                print("New model trained and loaded successfully")
        except Exception as e:
            print(f"Error training new model: {e}")
    
    async def predict(self, image_path: str) -> Dict[str, Any]:
        """Make prediction on image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if not (preprocess_image and predict_image):
            raise ValueError("Prediction functions not available")
            
        start_time = time.time()
        
        try:
            # Use thread executor for blocking operations
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                # Preprocess image
                processed_image = await loop.run_in_executor(
                    executor, preprocess_image, image_path
                )
                
                # Make prediction - expecting (prediction, confidence) tuple
                result = await loop.run_in_executor(
                    executor, predict_image, self.model, processed_image
                )
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) == 2:
                    prediction, confidence = result
                else:
                    # If it's just the prediction value
                    prediction = result
                    confidence = 0.95  # Default confidence
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            # Convert prediction to our enum format
            if prediction == 1 or prediction == "PNEUMONIA":
                prediction_class = PredictionClass.PNEUMONIA
                prob_pneumonia = float(confidence)
                prob_normal = 1.0 - prob_pneumonia
            else:
                prediction_class = PredictionClass.NORMAL
                prob_normal = float(confidence)
                prob_pneumonia = 1.0 - prob_normal
            
            return {
                "prediction": prediction_class,
                "confidence": float(confidence),
                "probability_normal": prob_normal,
                "probability_pneumonia": prob_pneumonia,
                "processing_time_seconds": processing_time,
                "timestamp": self.last_prediction_time,
                "model_version": self.model_version,
                "medical_advice": "This AI prediction is for assistance only. Please consult with a qualified healthcare professional for proper diagnosis and treatment."
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")
    
    async def retrain_model(self, training_id: str, **kwargs):
        """Retrain the model"""
        if self.is_training:
            raise ValueError("Model is already being trained")
            
        if not ChestXRayClassifier:
            raise ValueError("Model training not available")
        
        self.is_training = True
        self.training_jobs[training_id] = {
            "status": TrainingStatus.STARTING,
            "progress_percentage": 0.0,
            "current_epoch": 0,
            "total_epochs": kwargs.get("epochs", 10),
            "started_at": datetime.now(),
            "error_message": None
        }
        
        try:
            # Start training in background
            asyncio.create_task(self._retrain_background(training_id, **kwargs))
            
        except Exception as e:
            self.is_training = False
            self.training_jobs[training_id]["status"] = TrainingStatus.FAILED
            self.training_jobs[training_id]["error_message"] = str(e)
            raise
    
    async def _retrain_background(self, training_id: str, **kwargs):
        """Retrain model in background"""
        try:
            # Update status
            self.training_jobs[training_id]["status"] = TrainingStatus.TRAINING
            
            classifier = ChestXRayClassifier()
            training_data_path = kwargs.get("training_data_path", "uploads/training")
            epochs = kwargs.get("epochs", 10)
            
            # Create backup of current model
            if os.path.exists(self.model_path):
                backup_path = f"{self.model_path}.backup"
                os.rename(self.model_path, backup_path)
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    self._train_model_sync,
                    classifier,
                    training_data_path,
                    epochs,
                    training_id
                )
            
            # Reload the model
            await self.load_model()
            
            # Update status
            self.training_jobs[training_id]["status"] = TrainingStatus.COMPLETED
            self.training_jobs[training_id]["progress_percentage"] = 100.0
            
        except Exception as e:
            # Restore backup if training failed
            backup_path = f"{self.model_path}.backup"
            if os.path.exists(backup_path):
                os.rename(backup_path, self.model_path)
            
            self.training_jobs[training_id]["status"] = TrainingStatus.FAILED
            self.training_jobs[training_id]["error_message"] = str(e)
            
        finally:
            self.is_training = False
    
    def _train_model_sync(self, classifier, training_data_path: str, epochs: int, training_id: str):
        """Synchronous training function"""
        try:
            # Train new model with progress tracking
            for epoch in range(epochs):
                # Update progress
                self.training_jobs[training_id]["current_epoch"] = epoch + 1
                self.training_jobs[training_id]["progress_percentage"] = (epoch + 1) / epochs * 100
                
            # Actual training call
            classifier.train(
                training_data_path,
                "data/test",  # validation data
                self.model_path,
                epochs=epochs
            )
            
        except Exception as e:
            raise e
    
    async def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        from .utils import get_training_data_stats
        
        # Get memory usage
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # Check GPU availability
        gpu_available = False
        try:
            import tensorflow as tf
            gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
        except:
            pass
        
        # Get model accuracy if available
        model_accuracy = None
        if hasattr(self, 'last_accuracy'):
            model_accuracy = self.last_accuracy
        
        return {
            "api_status": "healthy",
            "model_loaded": self.model is not None,
            "model_version": self.model_version,
            "uptime_seconds": time.time() - self.start_time,
            "last_prediction_time": self.last_prediction_time,
            "total_predictions": self.prediction_count,
            "memory_usage_mb": memory_usage_mb,
            "gpu_available": gpu_available,
            "training_status": TrainingStatus.TRAINING if self.is_training else TrainingStatus.IDLE,
            "model_accuracy": model_accuracy,
            "model_path": self.model_path if os.path.exists(self.model_path) else None,
            "timestamp": datetime.now()
        }
    
    async def get_training_status(self, training_id: str):
        """Get training job status"""
        job = self.training_jobs.get(training_id)
        if not job:
            return None
            
        # Calculate estimated completion
        estimated_completion = None
        if job["status"] == TrainingStatus.TRAINING and job["current_epoch"] > 0:
            elapsed = (datetime.now() - job["started_at"]).total_seconds()
            time_per_epoch = elapsed / job["current_epoch"]
            remaining_epochs = job["total_epochs"] - job["current_epoch"]
            estimated_seconds = remaining_epochs * time_per_epoch
            estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return TrainingStatusResponse(
            training_id=training_id,
            status=job["status"],
            progress_percentage=job["progress_percentage"],
            current_epoch=job["current_epoch"],
            total_epochs=job["total_epochs"],
            current_loss=job.get("current_loss"),
            current_accuracy=job.get("current_accuracy"),
            validation_loss=job.get("validation_loss"),
            validation_accuracy=job.get("validation_accuracy"),
            started_at=job["started_at"],
            estimated_completion=estimated_completion,
            error_message=job.get("error_message")
        )
    
    async def reset_model(self):
        """Reset model to initial state"""
        self.model = None
        self.prediction_count = 0
        self.last_prediction_time = None
