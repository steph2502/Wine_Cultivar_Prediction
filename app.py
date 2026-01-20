from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        alcohol = float(request.form["alcohol"])
        malic_acid = float(request.form["malic_acid"])
        ash = float(request.form["ash"])
        alcalinity = float(request.form["alcalinity"])
        total_phenols = float(request.form["total_phenols"])
        flavanoids = float(request.form["flavanoids"])

        features = np.array([[alcohol, malic_acid, ash, alcalinity, total_phenols, flavanoids]])
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
