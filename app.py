from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
model_path = 'D:/Desktop/BrainTumor/braintumor.h5'  # Ensure this path is correct
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"The model file was not found at {model_path}")

# Define the labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Image decoding failed")
        return jsonify({'error': 'Image decoding failed'}), 400

    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({'prediction': predicted_label})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
