from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load your trained model (make sure this file is uploaded to Railway Files tab)
model_path = 'di_classification_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = load_model(model_path)

# Define the classes (update according to your model)
class_names = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # resize to match model input
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return "✅ Skin Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    try:
        img_tensor = preprocess_image(img_bytes)
        prediction = model.predict(img_tensor)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
