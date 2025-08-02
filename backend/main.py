"""
FastAPI main application for Chest X-ray Pneumonia Classification
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

from app.api import router as api_router
from app.models import ModelManager

# Global model manager instance
model_manager = None
app_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, app_start_time
    
    print("Starting Chest X-ray Classification API...")
    app_start_time = datetime.now()
    
    model_manager = ModelManager()
    
    try:
        await model_manager.load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load existing model: {e}")
        print("Model will be trained on first request")
    
    print("API is ready to serve requests!")
    
    yield
    
    print("Shutting down Chest X-ray Classification API...")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Chest X-ray Pneumonia Classification API",
    description="API for automated chest X-ray pneumonia detection using deep learning.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1", tags=["Chest X-ray Analysis"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Chest X-ray Pneumonia Classification API",
        "version": "1.0.0",
        "status": "healthy",
        "documentation": "/docs",
        "uptime_seconds": (datetime.now() - app_start_time).total_seconds() if app_start_time else 0,
        "endpoints": {
            "predict": "/api/v1/predict",
            "upload": "/api/v1/upload", 
            "retrain": "/api/v1/retrain",
            "status": "/api/v1/status"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_start_time).total_seconds() if app_start_time else 0
    }


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
