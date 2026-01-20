from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("model/wine_cultivar_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        alcohol = float(request.form["alcohol"])
        malic_acid = float(request.form["malic_acid"])
        ash = float(request.form["ash"])
        alcalinity = float(request.form["alcalinity"])
        flavanoids = float(request.form["flavanoids"])
        color_intensity = float(request.form["color_intensity"])

        features = np.array([[alcohol, malic_acid, ash, alcalinity, flavanoids, color_intensity]])
        features_scaled = scaler.transform(features)

        result = model.predict(features_scaled)[0]
        prediction = f"Cultivar {result + 1}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
