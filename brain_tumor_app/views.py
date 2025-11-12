from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from .forms import ImageUploadForm

# Load model once when server starts
model = None

def load_ml_model():
    global model
    if model is None:
        model = load_model(settings.ML_MODEL_PATH)
    return model

# Define class names matching your training data
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_tumor(request):
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
            
            # Load model
            model = load_ml_model()
            
            # Preprocess and predict
            processed_image = preprocess_image(file_path)
            predictions = model.predict(processed_image)
            
            # Get prediction results
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx] * 100)
            
            # Get all class probabilities
            all_predictions = {
                CLASS_NAMES[i]: float(predictions[0][i] * 100) 
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
