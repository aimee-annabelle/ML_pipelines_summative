# Chest X-ray Pneumonia Classification API

A FastAPI-based REST API for pneumonia detection in chest X-ray images using deep learning.

## Setup

1. Create virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
cd backend
python main.py
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /api/v1/predict` - Predict pneumonia from chest X-ray
- `POST /api/v1/upload` - Upload training images
- `POST /api/v1/retrain` - Retrain model
- `GET /api/v1/status` - Get model status

## Usage

Access the interactive API documentation at: http://localhost:8000/docs

The API accepts JPEG/PNG chest X-ray images and returns predictions with confidence scores.

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI application
│   └── app/
│       ├── api.py        # API routes
│       ├── models.py     # Data models and ML integration
│       └── utils.py      # Utility functions
├── src/                  # ML model code
├── data/                 # Training data
├── models/               # Trained models
└── requirements.txt      # Dependencies
```
