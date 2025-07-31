"""
loadModel.py

Author: Arjun Maganti
Date: 07-17-2025
Last Updated: 07-28-2025
"""
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.saving import register_keras_serializable
import warnings
warnings.filterwarnings('ignore')

# Register the custom resize_tensor function
@register_keras_serializable()
def resize_tensor(inputs):
    tensor_to_resize, reference_tensor = inputs
    # Cast to float32 to ensure consistency
    tensor_to_resize = tf.cast(tensor_to_resize, tf.float32)
    reference_shape = tf.shape(reference_tensor)[1:3]
    return tf.image.resize(tensor_to_resize, reference_shape)

# Define custom metrics and loss functions
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation"""
    # Ensure both tensors are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined loss: Categorical crossentropy + Dice loss"""
    # Ensure both tensors are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice_loss_val = dice_loss(y_true, y_pred)
    return ce_loss + dice_loss_val

def iou_score(y_true, y_pred):
    """IoU score for segmentation"""
    # Ensure both tensors are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return intersection / (union + 1e-6)

def load_deeplab_model(model_path):
    """
    Load the DeepLab model with all custom objects
    """
    # Set default dtype to float32
    tf.keras.backend.set_floatx('float32')

    # Define custom objects dictionary
    custom_objects = {
        'resize_tensor': resize_tensor,
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss,
        'iou_score': iou_score
    }

    # Load the model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )

    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    return model

def main(MODEL_PATH):
    model_path = MODEL_PATH
    model = load_deeplab_model(model_path)
    
    return model
