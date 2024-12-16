from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')  # Ensure 'index.html' is in 'templates' folder
CORS(app)

# Load the trained model
model = joblib.load("land_price_prediction_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Log input data
        
        # Create DataFrame from input data
        input_data = pd.DataFrame({
            "location": [float(data['location'])],
            "accessibility": [float(data['accessibility'])],
            "neighborhood_quality": [float(data['neighborhood_quality'])],
            "zoning": [int(data['zoning'])],
            "historical_trends": [float(data['historical_trends'])],
            "land_size": [float(data['land_size'])]
        })
        print("Input DataFrame for prediction:", input_data)  # Log DataFrame

        # Make prediction
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)  # Log prediction result

        return jsonify({
            'predicted_price': float(prediction[0]),
            'success': True
        })
    except Exception as e:
        print("Error during prediction:", e)  # Log error
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    try:
        feature_importance = model.feature_importances_
        features = ["location", "accessibility", "neighborhood_quality", "zoning", "historical_trends", "land_size"]
        
        importance_dict = {features[i]: float(feature_importance[i]) for i in range(len(features))}
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
