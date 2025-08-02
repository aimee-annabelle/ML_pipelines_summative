#!/usr/bin/env python3
"""
Chest X-ray Pneumonia Classification Training Script

This script trains a CNN model to classify chest X-rays as normal or pneumonia.
It includes data preprocessing, model training, evaluation, and saves the trained model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Add src directory to path
sys.path.append('src')

from model import ChestXrayClassifier
from preprocessing import display_sample_images


def train_xray_classifier():
    """
    Main function to train the chest X-ray classifier.
    """
    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA CLASSIFICATION")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Define paths
    train_dir = "data/train"
    test_dir = "data/test"
    models_dir = "models"
    model_save_path = os.path.join(models_dir, "xray_model.h5")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if data directories exist
    if not os.path.exists(train_dir):
        print(f"Error: Training data directory '{train_dir}' not found!")
        return None
    
    if not os.path.exists(test_dir):
        print(f"Error: Test data directory '{test_dir}' not found!")
        return None
    
    # Display data information
    print(f"\nData Information:")
    print(f"Training data: {train_dir}")
    print(f"Test data: {test_dir}")
    
    # Count training samples
    train_normal = len([f for f in os.listdir(os.path.join(train_dir, "NORMAL")) 
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    train_pneumonia = len([f for f in os.listdir(os.path.join(train_dir, "PNEUMONIA")) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    
    # Count test samples
    test_normal = len([f for f in os.listdir(os.path.join(test_dir, "NORMAL")) 
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    test_pneumonia = len([f for f in os.listdir(os.path.join(test_dir, "PNEUMONIA")) 
                         if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    
    print(f"Training samples - Normal: {train_normal}, Pneumonia: {train_pneumonia}")
    print(f"Test samples - Normal: {test_normal}, Pneumonia: {test_pneumonia}")
    print(f"Total training samples: {train_normal + train_pneumonia}")
    print(f"Total test samples: {test_normal + test_pneumonia}")
    
    # Display sample images
    print(f"\nDisplaying sample images...")
    try:
        display_sample_images(train_dir, num_samples=4)
    except Exception as e:
        print(f"Warning: Could not display sample images: {e}")
    
    # Create classifier - try custom CNN first
    print(f"\nCreating Custom CNN classifier...")
    classifier = ChestXrayClassifier(img_size=(224, 224), model_type="custom_cnn")
    
    # Build and display model architecture
    print(f"\nBuilding model architecture...")
    classifier.build_model()
    print(f"\nModel Summary:")
    classifier.model.summary()
    
    # Train model
    print(f"\nStarting model training...")
    print(f"This may take several minutes depending on your hardware...")
    
    try:
        history = classifier.train_model(
            train_dir=train_dir,
            epochs=15,  # Reduced for faster training
            batch_size=32,
            validation_split=0.2
        )
        
        print(f"\nTraining completed successfully!")
        
        # Plot training history
        print(f"\nPlotting training history...")
        try:
            classifier.plot_training_history()
        except Exception as e:
            print(f"Warning: Could not plot training history: {e}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
    # Evaluate model
    print(f"\nEvaluating model on test data...")
    try:
        results = classifier.evaluate_model(test_dir, batch_size=32)
        
        # Print detailed results
        print(f"\n" + "="*50)
        print(f"FINAL MODEL EVALUATION RESULTS")
        print(f"="*50)
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(f"-"*40)
        class_names = ['Normal', 'Pneumonia']
        print(classification_report(
            results['true_labels'], 
            results['predicted_classes'], 
            target_names=class_names
        ))
        
        print(f"\nConfusion Matrix:")
        print(f"-"*20)
        cm = results['confusion_matrix']
        print(f"               Predicted")
        print(f"             Normal  Pneumonia")
        print(f"Actual Normal    {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"     Pneumonia   {cm[1,0]:4d}      {cm[1,1]:4d}")
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nAdditional Metrics:")
        print(f"-"*20)
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        
        # Plot confusion matrix
        try:
            classifier.plot_confusion_matrix(results['confusion_matrix'])
        except Exception as e:
            print(f"Warning: Could not plot confusion matrix: {e}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        results = None
    
    # Save model
    print(f"\nSaving model...")
    try:
        classifier.save_model(model_save_path)
        print(f"Model successfully saved to: {model_save_path}")
        
        # Verify model file exists
        if os.path.exists(model_save_path):
            model_size = os.path.getsize(model_save_path) / (1024 * 1024)  # Size in MB
            print(f"Model file size: {model_size:.2f} MB")
        
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print(f"\n" + "="*60)
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved at: {model_save_path}")
    print(f"="*60)
    
    return classifier, results


def test_single_prediction(classifier, test_image_path=None):
    """
    Test the model with a single image prediction.
    
    Args:
        classifier: Trained classifier object
        test_image_path: Path to test image (optional)
    """
    if test_image_path is None:
        # Try to find a test image
        test_normal_dir = "data/test/NORMAL"
        test_pneumonia_dir = "data/test/PNEUMONIA"
        
        if os.path.exists(test_normal_dir):
            test_files = [f for f in os.listdir(test_normal_dir) 
                         if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if test_files:
                test_image_path = os.path.join(test_normal_dir, test_files[0])
        
        if test_image_path is None and os.path.exists(test_pneumonia_dir):
            test_files = [f for f in os.listdir(test_pneumonia_dir) 
                         if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if test_files:
                test_image_path = os.path.join(test_pneumonia_dir, test_files[0])
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"\nTesting single image prediction...")
        print(f"Test image: {test_image_path}")
        
        try:
            prediction, predicted_class, confidence = classifier.predict_single_image(test_image_path)
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"Raw prediction score: {prediction:.4f}")
        except Exception as e:
            print(f"Error during single prediction: {e}")


if __name__ == "__main__":
    # Train the model
    classifier, results = train_xray_classifier()
    
    # Test single prediction if training was successful
    if classifier is not None:
        test_single_prediction(classifier)
    else:
        print("Training failed. Please check the error messages above.")
