from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# Load the trained model
model = joblib.load("land_price_prediction_model.pkl")

# -----------------------
# Load dataset for metrics
# -----------------------
try:
    dataset = pd.read_csv("synthetic_land_data.csv")   # your synthetic dataset
    feature_cols = ["location", "accessibility", "neighborhood_quality", 
                    "zoning", "historical_trends", "land_size"]
    X = dataset[feature_cols]
    y_true = dataset["land_price"]   # assuming 'price' column exists

    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    model_metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "mse": float(mse)
    }
    print("Model metrics (evaluated on dataset):", model_metrics)

except Exception as e:
    print("Error loading dataset or computing metrics:", e)
    model_metrics = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame({
            "location": [float(data['location'])],
            "accessibility": [float(data['accessibility'])],
            "neighborhood_quality": [float(data['neighborhood_quality'])],
            "zoning": [int(data['zoning'])],
            "historical_trends": [float(data['historical_trends'])],
            "land_size": [float(data['land_size'])]
        })

        prediction = model.predict(input_data)
        pred_price = float(prediction[0])

        return jsonify({
            'predicted_price': pred_price,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    try:
        feature_importance = model.feature_importances_
        features = ["location", "accessibility", "neighborhood_quality", 
                    "zoning", "historical_trends", "land_size"]
        importance_dict = {features[i]: float(feature_importance[i]) for i in range(len(features))}
        return jsonify({'feature_importance': importance_dict, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

# -----------------------
# New route: get metrics
# -----------------------
@app.route('/metrics', methods=['GET'])
def metrics():
    if model_metrics:
        return jsonify({'success': True, 'metrics': model_metrics})
    else:
        return jsonify({'success': False, 'error': "Metrics unavailable"})

if __name__ == '__main__':
    app.run(debug=True)
