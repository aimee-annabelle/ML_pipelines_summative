"""
Prediction module for chest X-ray pneumonia classification.

This module contains functions for making predictions using trained deep learning models.
"""

import pandas as pd
import numpy as np
import os
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Model loading will fail.")

try:
    from preprocessing import preprocess_for_prediction
except ImportError:
    try:
        from .preprocessing import preprocess_for_prediction
    except ImportError:
        print("Warning: Could not import preprocessing functions")
        preprocess_for_prediction = None


def load_model(model_path):
    """
    Load a trained Keras model from file.
    
    Args:
        model_path (str): Path to the saved .h5 model file
        
    Returns:
        Loaded Keras model object
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for model loading")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = keras_load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {e}")


def predict_image(model, processed_image):
    """
    Make prediction on a preprocessed image using the loaded model.
    
    Args:
        model: Trained Keras model object
        processed_image: Preprocessed image array
        
    Returns:
        tuple: (prediction_class, confidence) where prediction_class is 0 (NORMAL) or 1 (PNEUMONIA)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for predictions")
    
    try:
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Handle different model output formats
        if prediction.shape[-1] == 1:
            # Binary classification with single output (sigmoid)
            confidence = float(prediction[0][0])
            prediction_class = 1 if confidence > 0.5 else 0
            if prediction_class == 0:
                confidence = 1 - confidence  # Adjust confidence for NORMAL class
        else:
            # Binary classification with two outputs (softmax)
            prediction_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))
        
        return prediction_class, confidence
        
    except Exception as e:
        raise Exception(f"Prediction failed: {e}")


def predict_from_image_path(model_path, image_path):
    """
    Complete prediction pipeline from image path.
    
    Args:
        model_path (str): Path to the saved model
        image_path (str): Path to the image file
        
    Returns:
        dict: Prediction results with class and confidence
    """
    # Load model
    model = load_model(model_path)
    
    # Preprocess image
    if preprocess_for_prediction is None:
        raise ImportError("Preprocessing function not available")
    
    processed_image = preprocess_for_prediction(image_path)
    
    # Make prediction
    prediction_class, confidence = predict_image(model, processed_image)
    
    # Convert to readable format
    class_name = "PNEUMONIA" if prediction_class == 1 else "NORMAL"
    
    return {
        "prediction": class_name,
        "prediction_class": prediction_class,
        "confidence": confidence,
        "image_path": image_path
    }

def predict_from_file(model_path, data_path, output_path=None):
    """
    Legacy function - not used for image classification.
    Kept for backward compatibility.
    """
    raise NotImplementedError("Use predict_from_image_path for image classification")


def batch_predict(model_path, input_folder, output_folder):
    """
    Make predictions on multiple image files in a folder.
    
    Args:
        model_path (str): Path to the saved model
        input_folder (str): Path to folder containing image files
        output_folder (str): Path to folder for saving predictions
    """
    import os
    import json
    
    model = load_model(model_path)
    results = []
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_folder, filename)
            
            try:
                result = predict_from_image_path(model_path, input_path)
                result['filename'] = filename
                results.append(result)
                print(f"Processed {filename}: {result['prediction']} ({result['confidence']:.3f})")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results.append({
                    'filename': filename,
                    'error': str(e)
                })
    
    # Save results
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, 'batch_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Batch predictions saved to {output_file}")
    return results


if __name__ == "__main__":
    # Example usage
    print("Chest X-ray prediction module loaded successfully!")
    if TENSORFLOW_AVAILABLE:
        print("TensorFlow is available for model loading.")
    else:
        print("Warning: TensorFlow is not available.")
