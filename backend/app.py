import io
import os
import cv2
import numpy as np
import pydicom
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "custom_pneumonia_cnn_best.keras")
CLINICAL_THRESHOLD = float(os.environ.get("CLINICAL_THRESHOLD", "0.38"))
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Normal", "Lung Opacity / Pneumonia"]

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

def preprocess_image(file_bytes, filename):
    filename_lower = filename.lower()
    if filename_lower.endswith('.dcm'):
        dicom_data = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = dicom_data.pixel_array.astype(np.float32)
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-7) * 255.0
        img = pixel_array.astype(np.uint8)
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
        img = np.array(image)
    resized = cv2.resize(img, IMAGE_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    expected_channels = model.input_shape[-1]
    if expected_channels == 3:
        input_tensor = np.stack([normalized] * 3, axis=-1)
    else:
        input_tensor = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(input_tensor, axis=0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under key 'file'"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        file_bytes = file.read()
        tensor = preprocess_image(file_bytes, file.filename)
        preds = model.predict(tensor)
        if preds.shape[-1] == 1:
            prob_pneumonia = float(preds[0][0])
        else:
            prob_pneumonia = float(preds[0][1])
        prob_normal = 1.0 - prob_pneumonia
        pred_class_idx = 1 if prob_pneumonia >= CLINICAL_THRESHOLD else 0
        predicted_label = CLASS_NAMES[pred_class_idx]
        return jsonify({
            "predicted_class": predicted_label,
            "class_index": pred_class_idx,
            "pneumonia_probability": prob_pneumonia,
            "normal_probability": prob_normal,
            "operating_threshold": CLINICAL_THRESHOLD,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
