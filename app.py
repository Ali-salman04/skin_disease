from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained model
model = load_model('EN0_model.pkl')

# Define the classes (modify based on your model)
class_names = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # adjust based on your model
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return "Skin Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img_tensor = preprocess_image(img_bytes)

    prediction = model.predict(img_tensor)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
