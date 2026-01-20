from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Path to model folder
MODEL_PATH = os.path.join("model", "wine_cultivar_model.pkl")

# Load model and scaler safely
try:
    with open(MODEL_PATH, "rb") as file:
        model, scaler = pickle.load(file)
    print("✓ Model and scaler loaded successfully!")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}. Check your folder structure.")
    model, scaler = None, None

# Home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Check server configuration."}), 500

    try:
        # Read input from form
        alcohol = float(request.form["alcohol"])
        malic_acid = float(request.form["malic_acid"])
        ash = float(request.form["ash"])
        alcalinity = float(request.form["alcalinity"])
        flavanoids = float(request.form["flavanoids"])
        color_intensity = float(request.form["color_intensity"])

        # Create features array
        features = np.array([[alcohol, malic_acid, ash, alcalinity, flavanoids, color_intensity]])
        features_scaled = scaler.transform(features)

        # Predict
        pred_class = model.predict(features_scaled)[0]
        prediction = f"Cultivar {pred_class + 1}"  # +1 to match UCI dataset classes

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# Run server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
