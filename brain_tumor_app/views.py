from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from .forms import ImageUploadForm

# Load model once when server starts
interpreter = None
input_details = None
output_details = None

def load_ml_model():
    """Load TensorFlow Lite model and allocate tensors"""
    global interpreter, input_details, output_details
    if interpreter is None:
        interpreter = tf.lite.Interpreter(model_path=settings.ML_MODEL_PATH)
        interpreter.allocate_tensors()

        # Get input and output details once at startup
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

# Define class names matching your training data
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image_for_tflite(img_path, target_size=(150, 150)):
    """
    Preprocess image for TFLite model prediction

    Args:
        img_path: Path to the image file
        target_size: Target size for resizing (width, height)

    Returns:
        numpy array with shape (1, height, width, channels) and dtype float32
    """
    # Load and resize image
    img = Image.open(img_path).resize(target_size)

    # Convert to numpy array with float32 dtype (TFLite requirement)
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize to [0, 1] range
    img_array = img_array / 255.0

    return img_array

def predict_tflite(img_path, interpreter, input_details, output_details):
    """
    Run TFLite model inference on an image

    Args:
        img_path: Path to the image file
        interpreter: TFLite interpreter instance
        input_details: Model input details
        output_details: Model output details

    Returns:
        tuple: (predicted_label, predictions_array)
    """
    # Preprocess image
    input_data = preprocess_image_for_tflite(img_path)

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]

    # Get predicted class
    predicted_class_idx = np.argmax(predictions)
    predicted_label = CLASS_NAMES[predicted_class_idx]

    return predicted_label, predictions

def predict_tumor(request):
    """Handle brain tumor prediction requests"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get uploaded image
            uploaded_file = request.FILES['image']

            # Save image to media folder
            file_name = default_storage.save(
                f'uploads/{uploaded_file.name}',
                uploaded_file
            )
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)

            # Load TFLite model (cached after first load)
            interpreter, input_details, output_details = load_ml_model()

            # Run prediction
            predicted_class, predictions = predict_tflite(
                file_path,
                interpreter,
                input_details,
                output_details
            )

            # Calculate confidence
            confidence = float(predictions[np.argmax(predictions)] * 100)

            # Get all class probabilities
            all_predictions = {
                CLASS_NAMES[i]: float(predictions[i] * 100)
                for i in range(len(CLASS_NAMES))
            }

            # Get image URL for display
            file_url = settings.MEDIA_URL + file_name

            context = {
                'form': form,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'image_url': file_url,
                'has_prediction': True
            }

            return render(request, 'brain_tumor_app/result.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'brain_tumor_app/upload.html', {'form': form})
