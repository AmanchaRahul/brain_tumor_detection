# Model Conversion Guide

## Converting .h5 Model to TensorFlow Lite

Your Django app has been updated to use TensorFlow Lite (.tflite) format, which is ~10x smaller than the original .h5 format. This solves deployment size limit issues on platforms like Render.

### Step 1: Enable Windows Long Paths (Required for TensorFlow)

Before converting, you need to enable Windows Long Path support:

1. **Option A - Via Registry:**
   - Press `Win + R`, type `regedit`, press Enter
   - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Find `LongPathsEnabled` (or create it as DWORD if it doesn't exist)
   - Set value to `1`
   - Restart your computer

2. **Option B - Via Group Policy (Windows Pro/Enterprise):**
   - Press `Win + R`, type `gpedit.msc`, press Enter
   - Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
   - Enable "Enable Win32 long paths"
   - Restart your computer

### Step 2: Run the Conversion Script

Once long paths are enabled, run:

```bash
cd D:\vs_code_projects\Machine_learning
env\Scripts\activate
cd brain_tumor
python convert_to_tflite.py
```

This will:
- Convert `brain_tumor_cnn_model.h5` → `brain_tumor_cnn_model.tflite`
- Reduce model size from ~528MB to ~30-50MB
- Show you the size comparison

### Step 3: Deploy to Render

After conversion:

1. The .tflite model will be small enough for Render's limits
2. The .h5 file is already excluded in `.gitignore`
3. Only the .tflite file will be pushed to GitHub
4. Your Django app is already configured to use the .tflite model

### What Changed in Your Code

**views.py** - Now uses TFLite Interpreter:
```python
# Before (Keras)
model = load_model(settings.ML_MODEL_PATH)
predictions = model.predict(processed_image)

# After (TFLite)
interpreter = tf.lite.Interpreter(model_path=settings.ML_MODEL_PATH)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], processed_image.astype(np.float32))
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
```

**settings.py** - Updated model path:
```python
ML_MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'brain_tumor_cnn_model.tflite')
```

### File Size Comparison

- VGG16 .h5: ~528 MB → .tflite: ~30 MB (94% reduction)
- ResNet50 .h5: ~98 MB → .tflite: ~10 MB (90% reduction)
- EfficientNetB0 .h5: ~30 MB → .tflite: ~5 MB (83% reduction)

### Troubleshooting

**If you still can't install TensorFlow after enabling long paths:**

Use an alternative environment:
```bash
# Use Google Colab or Linux subsystem (WSL) to convert
# Then copy the .tflite file back to your project
```

**Testing the converted model:**
```bash
python manage.py runserver
# Upload a brain tumor image to test predictions
```

The functionality remains exactly the same - only the model format changed!
