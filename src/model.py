"""
CNN Model for chest X-ray pneumonia classification.

This module contains the CNN model architecture and training functions
for classifying chest X-rays as normal or pneumonia.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import create_data_generators, create_test_generator


class ChestXrayClassifier:
    """
    CNN model for chest X-ray pneumonia classification.
    """
    
    def __init__(self, img_size=(224, 224), model_type="custom_cnn"):
        """
        Initialize the classifier.
        
        Args:
            img_size (tuple): Input image size (height, width)
            model_type (str): Type of model ('custom_cnn', 'mobilenet', 'resnet50')
        """
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def create_custom_cnn(self):
        """
        Create a custom CNN architecture for chest X-ray classification.
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 1)),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Fourth convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_mobilenet_model(self):
        """
        Create a MobileNetV2-based model for transfer learning.
        
        Returns:
            tf.keras.Model: Compiled MobileNetV2 model
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = Sequential([
            # Convert grayscale to RGB by repeating channels
            tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_resnet_model(self):
        """
        Create a ResNet50-based model for transfer learning.
        
        Returns:
            tf.keras.Model: Compiled ResNet50 model
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = Sequential([
            # Convert grayscale to RGB by repeating channels
            tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)),
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_model(self):
        """
        Build the specified model architecture.
        """
        if self.model_type == "custom_cnn":
            self.model = self.create_custom_cnn()
        elif self.model_type == "mobilenet":
            self.model = self.create_mobilenet_model()
        elif self.model_type == "resnet50":
            self.model = self.create_resnet_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Model created: {self.model_type}")
        return self.model
    
    def train_model(self, train_dir, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train the model using data generators.
        
        Args:
            train_dir (str): Path to training data directory
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            validation_split (float): Fraction of data for validation
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_generator, validation_generator = create_data_generators(
            train_dir, batch_size, self.img_size, validation_split
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/best_xray_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, test_dir, batch_size=32):
        """
        Evaluate the model on test data and generate comprehensive metrics.
        
        Args:
            test_dir (str): Path to test data directory
            batch_size (int): Batch size for evaluation
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Create test generator
        test_generator = create_test_generator(test_dir, batch_size, self.img_size)
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_classes)
        precision = precision_score(true_labels, predicted_classes)
        recall = recall_score(true_labels, predicted_classes)
        f1 = f1_score(true_labels, predicted_classes)
        
        # Generate classification report
        class_names = ['Normal', 'Pneumonia']
        report = classification_report(
            true_labels, predicted_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        
        # Create results dictionary
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_labels': true_labels
        }
        
        # Print results
        print(f"\n=== Model Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(true_labels, predicted_classes, target_names=class_names))
        
        return results
    
    def plot_confusion_matrix(self, confusion_matrix, class_names=['Normal', 'Pneumonia']):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix from evaluation
            class_names: List of class names
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix - Chest X-ray Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history (loss and accuracy curves).
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
    
    def predict_single_image(self, img_path):
        """
        Predict pneumonia for a single chest X-ray image.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            tuple: (prediction_probability, predicted_class, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        from preprocessing import preprocess_for_prediction
        
        # Preprocess image
        img = preprocess_for_prediction(img_path, self.img_size)
        if img is None:
            return None, None, None
        
        # Make prediction
        prediction = self.model.predict(img, verbose=0)[0][0]
        predicted_class = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return prediction, predicted_class, confidence


def main():
    """
    Main function to train and evaluate the chest X-ray classification model.
    """
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Define paths
    train_dir = "data/train"
    test_dir = "data/test"
    model_save_path = "models/xray_model.h5"
    
    # Create classifier
    print("Creating chest X-ray classifier...")
    classifier = ChestXrayClassifier(img_size=(224, 224), model_type="custom_cnn")
    
    # Build model
    classifier.build_model()
    classifier.model.summary()
    
    # Train model
    print("\nTraining model...")
    history = classifier.train_model(
        train_dir=train_dir,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    print("\nEvaluating model...")
    results = classifier.evaluate_model(test_dir, batch_size=32)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    classifier.save_model(model_save_path)
    
    print(f"\nTraining completed! Model saved to: {model_save_path}")
    
    return classifier, results


if __name__ == "__main__":
    classifier, results = main()
