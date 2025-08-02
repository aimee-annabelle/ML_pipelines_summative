# ML Pipelines Summative Project - Chest X-ray Pneumonia Classification

## 📋 Project Description

This project implements a complete machine learning pipeline for **Chest X-ray Pneumonia Detection**, distinguishing between normal and pneumonia cases in chest radiographs. The system features:

- **Medical Image Classification**: Deep learning model trained to identify pneumonia in chest X-ray images
- **End-to-End ML Pipeline**: From medical image preprocessing to model deployment
- **FastAPI Backend**: RESTful API for medical image upload and real-time pneumonia prediction
- **Production-Ready Architecture**: Clean, maintainable code with proper validation and data handling

**Medical Use Case**: This system assists healthcare professionals in rapid pneumonia screening, providing AI-powered diagnostic support for chest X-ray interpretation.

The project showcases modern MLOps practices applied to medical imaging with a focus on scalability and maintainability.

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Git

### Python + FastAPI Backend Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/aimee-annabelle/ML_pipelines_summative.git
   cd ML_pipelines_summative
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv

   # Windows (Command Prompt)
   venv\Scripts\activate

   # Windows (PowerShell)
   venv\Scripts\Activate.ps1

   # Windows (Git Bash)
   source venv/Scripts/activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**

   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **API Documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📊 Dataset Information - Chest X-ray Pneumonia Classification

### Dataset Overview

The project uses a comprehensive chest X-ray dataset for pneumonia classification:

- **Total Images**: 5,856 chest X-ray images
- **Image Format**: JPEG
- **Classes**: 2 (Normal, Pneumonia)
- **Dataset Size**: ~1.3GB (managed with Git LFS)

### Dataset Distribution

| Split     | Normal | Pneumonia | Total |
| --------- | ------ | --------- | ----- |
| **Train** | 1,349  | 3,883     | 5,232 |
| **Test**  | 234    | 390       | 624   |
| **Total** | 1,583  | 4,273     | 5,856 |

### Class Distribution Analysis

- **Training Set**: 74.2% Pneumonia, 25.8% Normal
- **Test Set**: 62.5% Pneumonia, 37.5% Normal
- **Overall**: 72.9% Pneumonia, 27.1% Normal

### Technical Specifications

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow 2.19.0
- **Image Processing**: OpenCV 4.12.0
- **Input Format**: RGB chest X-ray images
- **Training Environment**: Python 3.12.8

## 🗂️ Project Structure

```
ML_pipelines_summative/
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── train_model.py         # Model training script
│
├── backend/               # FastAPI application
│   ├── main.py           # API entry point
│   └── app/
│       ├── api.py        # API routes
│       ├── models.py     # Data models and ML integration
│       └── utils.py      # Utility functions
│
├── src/                   # Core ML modules
│   ├── preprocessing.py   # Image preprocessing
│   ├── model.py          # CNN model definition
│   └── prediction.py     # Prediction pipeline
│
├── data/                  # Dataset (Git LFS)
│   ├── train/            # Training images (5,232)
│   │   ├── NORMAL/       # Normal X-rays (1,349)
│   │   └── PNEUMONIA/    # Pneumonia X-rays (3,883)
│   └── test/             # Test images (624)
│       ├── NORMAL/       # Normal X-rays (234)
│       └── PNEUMONIA/    # Pneumonia X-rays (390)
│
├── models/               # Trained model storage
├── notebook/             # Jupyter notebooks
│   └── project_name.ipynb
│
└── venv/                 # Virtual environment
```

## 🚀 API Endpoints

The FastAPI backend provides the following endpoints:

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - ReDoc API documentation

### ML Endpoints

- `POST /api/v1/predict` - Upload chest X-ray for pneumonia prediction
- `POST /api/v1/upload` - Upload training images
- `POST /api/v1/retrain` - Trigger model retraining
- `GET /api/v1/status` - Get current model status

### API Features

- **Async Operations**: Non-blocking image processing
- **File Validation**: JPEG/PNG format validation
- **Error Handling**: Comprehensive error responses
- **CORS Support**: Cross-origin requests enabled
- **Interactive Docs**: Built-in Swagger UI at `/docs`

## 🔧 Technology Stack

### Backend

- **FastAPI**: Modern async web framework
- **TensorFlow 2.19.0**: Deep learning framework
- **OpenCV 4.12.0**: Computer vision library
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server

### Data Management

- **Git LFS**: Large file storage for dataset
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Pillow**: Image processing

### Development

- **Python 3.12.8**: Programming language
- **Virtual Environment**: Isolated dependencies

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Aimee Annabelle**

- GitHub: [@aimee-annabelle](https://github.com/aimee-annabelle)

---

⭐ **Star this repository if you found it helpful!**
