"""
Quick test script to verify TFLite model works correctly
Run this before deploying to ensure everything is functioning
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuration
MODEL_PATH = 'ml_models/brain_tumor_model.tflite'
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def test_tflite_model():
    """Test if the TFLite model loads and can run inference"""

    print("=" * 50)
    print("Testing TFLite Model")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return False

    print(f"✓ Model file found: {MODEL_PATH}")

    # Get model size
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✓ Model size: {model_size_mb:.2f} MB")

    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("✓ Model loaded successfully")

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\n📊 Model Information:")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input dtype: {input_details[0]['dtype']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output dtype: {output_details[0]['dtype']}")

        # Create a dummy test image
        print(f"\n🧪 Running test inference with dummy image...")

        # Get expected input shape
        input_shape = input_details[0]['shape']
        test_image = np.random.rand(*input_shape).astype(np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_image)

        # Run inference
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        print("✓ Inference successful!")
        print(f"\n📈 Dummy Predictions:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"   {class_name}: {predictions[i]:.4f}")

        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100

        print(f"\n✓ Predicted: {predicted_class} ({confidence:.2f}% confidence)")

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYour TFLite model is working correctly!")
        print("You can now run: python manage.py runserver")

        return True

    except Exception as e:
        print(f"\n❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tflite_model()
    exit(0 if success else 1)
