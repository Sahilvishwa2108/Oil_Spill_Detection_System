"""
Model Compatibility Fix Script
This script creates simple placeholder models that are compatible with TensorFlow 2.15
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

def create_simple_unet_model(input_shape=(256, 256, 3)):
    """Create a simple U-Net-like model for oil spill detection"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = tf.keras.layers.concatenate([up1, conv2])
    conv4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    conv4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
    
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = tf.keras.layers.concatenate([up2, conv1])
    conv5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(up2)
    conv5 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv5)
    
    # Output layer for binary classification
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_simple_deeplab_model(input_shape=(256, 256, 3)):
    """Create a simple DeepLab-like model for oil spill detection"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Simple CNN backbone
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    
    # Global average pooling for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_compatible_models():
    """Create and save compatible models"""
    print("Creating compatible models for TensorFlow 2.15...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create and save U-Net model
    print("Creating U-Net model...")
    unet_model = create_simple_unet_model()
    unet_model.save("models/unet_final_model_compatible.h5")
    print("✅ U-Net model saved")
    
    # Create and save DeepLab model
    print("Creating DeepLab model...")
    deeplab_model = create_simple_deeplab_model()
    deeplab_model.save("models/deeplab_final_model_compatible.h5")
    print("✅ DeepLab model saved")
    
    print("✅ Compatible models created successfully!")

if __name__ == "__main__":
    create_compatible_models()
