from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# Load model, scaler, and feature order
model = joblib.load("improved_land_price_model.pkl")
scaler = joblib.load("feature_scaler.pkl")
FEATURE_ORDER = joblib.load("feature_order.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create DataFrame with base features
        input_data = pd.DataFrame({
            "location": [float(data['location'])],
            "accessibility": [float(data['accessibility'])],
            "neighborhood_quality": [float(data['neighborhood_quality'])],
            "zoning": [int(data['zoning'])],
            "historical_trends": [float(data['historical_trends'])],
            "land_size": [float(data['land_size'])]
        })
        
        # Add engineered features
        input_data['location_accessibility'] = input_data['location'] * input_data['accessibility']
        input_data['neighborhood_zoning'] = input_data['neighborhood_quality'] * (input_data['zoning'] == 2)
        input_data['log_land_size'] = np.log(input_data['land_size'])
        
        # Ensure correct feature order
        input_data = input_data[FEATURE_ORDER]
        
        # Scale features
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        
        return jsonify({
            'predicted_price': float(prediction[0]),
            'success': True
        })
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    try:
        feature_importance = model.feature_importances_
        importance_dict = {FEATURE_ORDER[i]: float(feature_importance[i]) 
                         for i in range(len(FEATURE_ORDER))}
        return jsonify({
            'feature_importance': importance_dict,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    app.run(debug=True)