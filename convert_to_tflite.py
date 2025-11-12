"""
Script to convert .h5 model to TensorFlow Lite format
This reduces model size from ~500MB to ~30MB for deployment
"""

import tensorflow as tf
import os

# Paths
H5_MODEL_PATH = 'ml_models/brain_tumor_cnn_model.h5'
TFLITE_MODEL_PATH = 'ml_models/brain_tumor_cnn_model.tflite'

def convert_h5_to_tflite():
    """Convert Keras .h5 model to TensorFlow Lite format"""

    print(f"Loading model from: {H5_MODEL_PATH}")

    # Load the .h5 model
    model = tf.keras.models.load_model(H5_MODEL_PATH)

    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Convert to TFLite
    print("\nConverting to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional: Apply optimizations to reduce size further
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the TFLite model
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    # Get file sizes
    h5_size = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)  # MB

    print(f"\n✓ Conversion successful!")
    print(f"Original .h5 model size: {h5_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")
    print(f"Size reduction: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
    print(f"\nTFLite model saved to: {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    convert_h5_to_tflite()
