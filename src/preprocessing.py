"""
Image preprocessing module for chest X-ray classification.

This module contains functions for loading, preprocessing, and augmenting
chest X-ray images for pneumonia detection.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some functions may be limited.")
import matplotlib.pyplot as plt


def load_and_preprocess_images(data_dir, img_size=(224, 224)):
    """
    Load and preprocess chest X-ray images from directory structure.
    
    Args:
        data_dir (str): Path to data directory containing NORMAL and PNEUMONIA folders
        img_size (tuple): Target image size (height, width)
        
    Returns:
        tuple: (images array, labels array)
    """
    images = []
    labels = []
    
    # Load Normal images (label 0)
    normal_dir = os.path.join(data_dir, 'NORMAL')
    if os.path.exists(normal_dir):
        for filename in os.listdir(normal_dir):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(normal_dir, filename)
                img = load_and_process_image(img_path, img_size)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # Normal = 0
    
    # Load Pneumonia images (label 1)
    pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        for filename in os.listdir(pneumonia_dir):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(pneumonia_dir, filename)
                img = load_and_process_image(img_path, img_size)
                if img is not None:
                    images.append(img)
                    labels.append(1)  # Pneumonia = 1
    
    return np.array(images), np.array(labels)


def load_and_process_image(img_path, img_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        img_path (str): Path to image file
        img_size (tuple): Target image size
        
    Returns:
        np.array: Preprocessed image array
    """
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize image
        img = cv2.resize(img, img_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension for grayscale
        img = np.expand_dims(img, axis=-1)
        
        return img
        
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None


def create_data_generators(train_dir, batch_size=32, img_size=(224, 224), validation_split=0.2):
    """
    Create data generators with augmentation for training and validation.
    
    Args:
        train_dir (str): Path to training data directory
        batch_size (int): Batch size for training
        img_size (tuple): Target image size
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        subset='training',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator


def create_test_generator(test_dir, batch_size=32, img_size=(224, 224)):
    """
    Create test data generator without augmentation.
    
    Args:
        test_dir (str): Path to test data directory
        batch_size (int): Batch size for testing
        img_size (tuple): Target image size
        
    Returns:
        test_generator: Test data generator
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    return test_generator


def preprocess_for_prediction(img_path, img_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    
    Args:
        img_path (str): Path to image file
        img_size (tuple): Target image size
        
    Returns:
        np.array: Preprocessed image ready for prediction
    """
    img = load_and_process_image(img_path, img_size)
    if img is not None:
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    return None


def display_sample_images(data_dir, num_samples=4):
    """
    Display sample images from the dataset.
    
    Args:
        data_dir (str): Path to data directory
        num_samples (int): Number of samples to display per class
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    # Display Normal images
    normal_dir = os.path.join(data_dir, 'NORMAL')
    normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    
    for i in range(min(num_samples, len(normal_files))):
        img_path = os.path.join(normal_dir, normal_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Normal {i+1}')
        axes[0, i].axis('off')
    
    # Display Pneumonia images
    pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
    pneumonia_files = [f for f in os.listdir(pneumonia_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    
    for i in range(min(num_samples, len(pneumonia_files))):
        img_path = os.path.join(pneumonia_dir, pneumonia_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Pneumonia {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully!")
