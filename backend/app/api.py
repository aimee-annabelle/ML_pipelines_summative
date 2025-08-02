from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
from typing import List
import tempfile
import os
from datetime import datetime
import logging

from .models import (
    ModelManager, PredictionResponse, UploadResponse, RetrainResponse, 
    StatusResponse, PredictionClass, TrainingStatus, TrainingParameters,
    UploadedFile
)
from .utils import (
    validate_image, save_uploaded_file, get_training_data_stats,
    sanitize_filename
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


def get_model_manager():
    """Get the global model manager instance"""
    import sys
    try:
        return getattr(sys.modules.get('main'), 'model_manager', None)
    except Exception as e:
        logger.error(f"Failed to get model manager: {e}")
        return None


def get_app_start_time():
    """Get the app start time"""
    import sys
    try:
        return getattr(sys.modules.get('main'), 'app_start_time', None)
    except Exception as e:
        logger.error(f"Failed to get app start time: {e}")
        return None


@router.post("/predict", response_model=PredictionResponse)
async def predict_pneumonia(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG, PNG)")
):
    try:
        await validate_image(file)
        
        model_manager = get_model_manager()
        
        if not model_manager or not model_manager.model:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please wait for model initialization or check system status."
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            result = await model_manager.predict(tmp_file_path)
            
            logger.info(f"Prediction made for file: {file.filename}")
            
            return PredictionResponse(
                filename=file.filename,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probability_normal=result["probability_normal"],
                probability_pneumonia=result["probability_pneumonia"],
                processing_time_seconds=result["processing_time_seconds"],
                timestamp=result["timestamp"],
                medical_advice=result["medical_advice"],
                model_version=result["model_version"]
            )
            
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_training_images(
    files: List[UploadFile] = File(..., description="Multiple chest X-ray training images"),
    labels: str = Form(..., description="Comma-separated labels (NORMAL,PNEUMONIA,NORMAL,...)")
):
    try:
        label_list = [label.strip().upper() for label in labels.split(",")]
        
        if len(files) != len(label_list):
            raise HTTPException(
                status_code=400, 
                detail=f"Number of files ({len(files)}) must match number of labels ({len(label_list)})"
            )
        
        valid_labels = {"NORMAL", "PNEUMONIA"}
        for label in label_list:
            if label not in valid_labels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid label '{label}'. Must be one of: {valid_labels}"
                )
        
        uploaded_files = []
        for file, label in zip(files, label_list):
            await validate_image(file)
            saved_path = await save_uploaded_file(file, label)
            uploaded_files.append(UploadedFile(
                filename=sanitize_filename(file.filename),
                label=PredictionClass(label),
                saved_path=saved_path,
                size_bytes=file.size or 0
            ))
        
        logger.info(f"Uploaded {len(uploaded_files)} training images")
        
        stats = get_training_data_stats()
        
        return UploadResponse(
            message=f"Successfully uploaded {len(uploaded_files)} training images",
            uploaded_files=uploaded_files,
            total_files=len(uploaded_files),
            timestamp=datetime.now(),
            ready_for_training=stats["total_files"] >= 10
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_model(
    background_tasks: BackgroundTasks,
    use_uploaded_data: bool = Form(True, description="Include uploaded training data"),
    epochs: int = Form(5, description="Number of training epochs"),
    learning_rate: float = Form(0.001, description="Learning rate for training")
):
    try:
        model_manager = get_model_manager()
        
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not initialized")
        
        training_id = f"training_{int(datetime.now().timestamp())}"
        
        training_params = TrainingParameters(
            epochs=epochs,
            learning_rate=learning_rate,
            use_uploaded_data=use_uploaded_data
        )
        
        try:
            # Determine training data path
            training_data_path = "uploads/training"  # Default to uploaded data
            
            if not use_uploaded_data:
                # Check if original data folder exists (Git LFS or local)
                if os.path.exists("data/train") and os.path.exists("data/train/NORMAL") and os.path.exists("data/train/PNEUMONIA"):
                    # Verify the data folders have content
                    normal_count = len(os.listdir("data/train/NORMAL")) if os.path.exists("data/train/NORMAL") else 0
                    pneumonia_count = len(os.listdir("data/train/PNEUMONIA")) if os.path.exists("data/train/PNEUMONIA") else 0
                    
                    if normal_count > 0 and pneumonia_count > 0:
                        training_data_path = "data/train"
                        logger.info(f"Using original dataset: {normal_count} normal, {pneumonia_count} pneumonia images")
                    else:
                        logger.warning("Original dataset folders exist but appear empty")
                        raise HTTPException(
                            status_code=400, 
                            detail="Original training data is empty. Please upload training images first."
                        )
                else:
                    logger.warning("Original dataset not found")
                    raise HTTPException(
                        status_code=400, 
                        detail="Original training data not available. Please upload training images first."
                    )
            
            await model_manager.retrain_model(
                training_id=training_id,
                epochs=epochs,
                learning_rate=learning_rate,
                training_data_path=training_data_path
            )
        except Exception as e:
            logger.error(f"Failed to start retraining: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")
        
        logger.info(f"Started retraining job: {training_id}")
        
        return RetrainResponse(
            message="Model retraining started successfully",
            training_id=training_id,
            status=TrainingStatus.STARTING,
            estimated_completion_minutes=epochs * 2,
            parameters=training_params,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_model_status():
    """Get API and model status"""
    try:
        model_manager = get_model_manager()
        app_start_time = get_app_start_time()
        
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not initialized")
        
        status_data = await model_manager.get_status()
        
        return StatusResponse(
            api_status=status_data["api_status"],
            model_loaded=status_data["model_loaded"],
            model_version=status_data["model_version"],
            uptime_seconds=status_data["uptime_seconds"],
            last_prediction_time=status_data["last_prediction_time"],
            total_predictions=status_data["total_predictions"],
            memory_usage_mb=status_data["memory_usage_mb"],
            gpu_available=status_data["gpu_available"],
            training_status=status_data["training_status"],
            model_accuracy=status_data["model_accuracy"],
            model_path=status_data["model_path"],
            timestamp=status_data["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
