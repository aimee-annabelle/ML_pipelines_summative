# ML Pipelines Summative Project - Chest X-ray Pneumonia Classification

## 📋 Project Description

This project implements a complete machine learning pipeline for **Chest X-ray Pneumonia Detection**, distinguishing between normal and pneumonia cases in chest radiographs. The system features:

- **Medical Image Classification**: Deep learning model trained to identify pneumonia in chest X-ray images
- **End-to-End ML Pipeline**: From medical image preprocessing to model deployment in healthcare settings
- **FastAPI Backend**: RESTful API for secure medical image upload and real-time pneumonia prediction
- **React Frontend**: User-friendly medical interface for healthcare professionals to upload X-rays and view diagnostic results
- **Clinical-Grade Architecture**: Production-ready deployment with proper validation and medical data handling
- **Comprehensive Testing**: Including flood testing for performance validation in high-throughput clinical environments

**Medical Use Case**: This system assists healthcare professionals in rapid pneumonia screening, providing AI-powered diagnostic support for chest X-ray interpretation with high accuracy and reliability.

The project showcases modern MLOps practices applied to medical imaging with a focus on scalability, maintainability, and clinical user experience.

## 🎥 Demo

**YouTube Demo**: [Coming Soon - Demo will be uploaded here]

## 🚀 Live Applications

- **FastAPI Backend**: [Will be deployed and linked here]
- **React Frontend**: [Will be deployed and linked here]

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
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

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   # Create .env file
   echo "DATABASE_URL=sqlite:///./ml_app.db" > .env
   echo "SECRET_KEY=your-secret-key-here" >> .env
   ```

5. **Run the FastAPI server**

   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **API Documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Node + React Frontend Setup

1. **Navigate to frontend directory**

   ```bash
   cd frontend
   ```

2. **Install Node dependencies**

   ```bash
   npm install
   ```

3. **Set up environment variables**

   ```bash
   # Create .env file in frontend directory
   echo "REACT_APP_API_URL=http://localhost:8000" > .env
   echo "REACT_APP_VERSION=1.0.0" >> .env
   ```

4. **Start the development server**

   ```bash
   npm start
   ```

5. **Build for production**
   ```bash
   npm run build
   ```

### Docker Setup (Alternative)

1. **Build and run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

2. **Access the applications**
   - Backend: http://localhost:8000
   - Frontend: http://localhost:3000

## 🔥 Flood Test Results

_Performance testing will be conducted and results will be documented here once the application is fully deployed._

### Performance Metrics

| Metric                    | Result | Threshold   | Status  |
| ------------------------- | ------ | ----------- | ------- |
| **Requests per Second**   | TBD    | > 500 req/s | Pending |
| **Average Response Time** | TBD    | < 200ms     | Pending |
| **95th Percentile**       | TBD    | < 500ms     | Pending |
| **Error Rate**            | TBD    | < 1%        | Pending |
| **Concurrent Users**      | TBD    | > 500       | Pending |

### Load Testing Details

```bash
# Load testing will be performed using Artillery
# Results will be updated once testing is complete
```

### Resource Utilization

- **CPU Usage**: TBD
- **Memory Usage**: TBD
- **Database Connections**: TBD

## 📱 Screenshots

_Screenshots will be added here once the frontend and backend are fully implemented._

### Frontend Interface

#### Medical Dashboard

_Screenshot coming soon_
_Healthcare professional dashboard showing diagnostic statistics and patient queue_

#### X-ray Upload Interface

_Screenshot coming soon_
_Secure medical image upload interface with drag-and-drop functionality_

#### Pneumonia Detection Results

_Screenshot coming soon_
_Real-time pneumonia classification results with confidence scores and visual indicators_

#### Diagnostic History

_Screenshot coming soon_
_Patient diagnostic history and batch processing results for clinical workflow_

### Backend API Documentation

#### Medical API Documentation

_Screenshot coming soon_
_FastAPI documentation for medical image processing endpoints_

#### Prediction API Response

_Screenshot coming soon_
_Example JSON response from pneumonia detection endpoint with confidence metrics_

## 📊 Model Metrics - Chest X-ray Pneumonia Classification

_Model training will be performed and metrics will be documented here once the model is developed._

### Training Performance

| Metric                    | Training Set | Validation Set | Test Set |
| ------------------------- | ------------ | -------------- | -------- |
| **Accuracy**              | TBD          | TBD            | TBD      |
| **Precision (Pneumonia)** | TBD          | TBD            | TBD      |
| **Recall (Pneumonia)**    | TBD          | TBD            | TBD      |
| **F1-Score (Pneumonia)**  | TBD          | TBD            | TBD      |
| **Specificity (Normal)**  | TBD          | TBD            | TBD      |
| **AUC-ROC**               | TBD          | TBD            | TBD      |

### Clinical Performance Metrics

| Clinical Metric                     | Value | Clinical Significance                           |
| ----------------------------------- | ----- | ----------------------------------------------- |
| **Sensitivity**                     | TBD   | High detection rate for pneumonia cases         |
| **Specificity**                     | TBD   | Low false positive rate for normal cases        |
| **PPV (Positive Predictive Value)** | TBD   | Probability that positive prediction is correct |
| **NPV (Negative Predictive Value)** | TBD   | Probability that negative prediction is correct |

### Model Details

- **Architecture**: Convolutional Neural Network (CNN) with Transfer Learning
- **Base Model**: TBD (will be selected during development)
- **Input Image Size**: TBD
- **Training Samples**: TBD chest X-ray images
- **Test Samples**: TBD chest X-ray images
- **Data Augmentation**: TBD
- **Training Time**: TBD
- **Inference Time**: TBD per X-ray image

### Class Distribution

| Class         | Training Samples | Test Samples | Description              |
| ------------- | ---------------- | ------------ | ------------------------ |
| **Normal**    | TBD              | TBD          | Healthy chest X-rays     |
| **Pneumonia** | TBD              | TBD          | X-rays showing pneumonia |

### Confusion Matrix

```
Will be updated once model training is complete
```

## 🗂️ Project Structure

```
ML_pipelines_summative/
│
├── README.md
├── requirements.txt
│
├── backend/
│   ├── main.py
│   ├── models/
│   ├── routers/
│   └── utils/
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── build/
│
├── notebook/
│   └── chest_xray_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   └── trained_model.pkl
│
│
└── screenshots/
    ├── dashboard.png
    ├── prediction-interface.png
    └── api-response.png
```

## 🧪 Testing

### Run Backend Tests

```bash
cd backend
pytest tests/ -v --coverage
```

### Run Frontend Tests

```bash
cd frontend
npm test
```

### API Health Check

```bash
curl http://localhost:8000/health
```

## 🚀 Deployment

### Heroku (Backend)

```bash
heroku create your-app-name
git push heroku main
```

### Netlify (Frontend)

```bash
npm run build
# Upload build folder to Netlify
```

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
