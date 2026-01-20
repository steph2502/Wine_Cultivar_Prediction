from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessors
try:
    model = joblib.load('titanic_survival_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✓ Model and preprocessors loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please check server configuration.'
            }), 500
        
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Encode categorical variables
        sex_encoded = 1 if sex.lower() == 'male' else 0
        embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked.upper()]
        
        # Create DataFrame
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_encoded],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked_encoded]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'result': 'Survived' if prediction == 1 else 'Did Not Survive',
            'survival_probability': float(probability[1]),
            'death_probability': float(probability[0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
