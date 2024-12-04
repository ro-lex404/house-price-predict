from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # More specific CORS configuration

# Load the trained model
model = joblib.load("land_price_prediction_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create DataFrame from input data
        input_data = pd.DataFrame({
            "location": [float(data['location'])],
            "accessibility": [float(data['accessibility'])],
            "neighborhood_quality": [float(data['neighborhood_quality'])],
            "zoning": [int(data['zoning'])],
            "historical_trends": [float(data['historical_trends'])],
            "land_size": [float(data['land_size'])]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'predicted_price': float(prediction[0]),
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400  # Added proper error status code

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    try:
        feature_importance = model.feature_importances_
        features = ["location", "accessibility", "neighborhood_quality", 
                   "zoning", "historical_trends", "land_size"]
        
        importance_dict = {
            features[i]: float(feature_importance[i]) 
            for i in range(len(features))
        }
        
        return jsonify({
            'feature_importance': importance_dict,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400  # Added proper error status code

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Explicitly set port