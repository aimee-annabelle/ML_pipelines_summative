# Frontend for Chest X-ray Classification

This folder contains the frontend application for the Chest X-ray Pneumonia Classification system.

## Deployment

This frontend is deployed separately from the backend API and should not trigger backend rebuilds.

## API Integration

The frontend will connect to the deployed backend API at:

- Production API: `https://chest-xray-api-3u81.onrender.com`
- API Documentation: `https://chest-xray-api-3u81.onrender.com/docs`

## Endpoints

- `POST /api/v1/predict` - Upload chest X-ray for prediction
- `POST /api/v1/upload` - Upload training images
- `POST /api/v1/retrain` - Trigger model retraining
- `GET /api/v1/status` - Get model status
- `GET /health` - Health check
