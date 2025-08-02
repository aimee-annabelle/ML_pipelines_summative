"""
Prediction module for ML pipeline.

This module contains functions for making predictions using trained models.
"""

import pandas as pd
import numpy as np
import joblib
from src.preprocessing import preprocess_features


def load_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Loaded model object
    """
    return joblib.load(model_path)


def make_predictions(model, data):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Trained model object
        data: Input data for predictions
        
    Returns:
        predictions: Model predictions
    """
    return model.predict(data)


def predict_from_file(model_path, data_path, output_path=None):
    """
    Make predictions from a file and optionally save results.
    
    Args:
        model_path (str): Path to the saved model
        data_path (str): Path to the input data file
        output_path (str, optional): Path to save predictions
        
    Returns:
        pd.DataFrame: Data with predictions
    """
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess data
    data = pd.read_csv(data_path)
    processed_data = preprocess_features(data)
    
    # Make predictions
    predictions = make_predictions(model, processed_data)
    
    # Add predictions to dataframe
    result_df = data.copy()
    result_df['predictions'] = predictions
    
    # Save results if output path is provided
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return result_df


def batch_predict(model_path, input_folder, output_folder):
    """
    Make predictions on multiple files in a folder.
    
    Args:
        model_path (str): Path to the saved model
        input_folder (str): Path to folder containing input files
        output_folder (str): Path to folder for saving predictions
    """
    import os
    
    model = load_model(model_path)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"predictions_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            predict_from_file(model_path, input_path, output_path)


if __name__ == "__main__":
    # Example usage
    print("Prediction module loaded successfully!")
