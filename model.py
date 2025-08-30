import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Load Data
data = pd.read_csv("land_price_data.csv")

# Define feature order explicitly
FEATURE_ORDER = [
    "location", "accessibility", "neighborhood_quality", "zoning",
    "historical_trends", "land_size", "location_accessibility",
    "neighborhood_zoning", "log_land_size"
]

# Feature Engineering
data['location_accessibility'] = data['location'] * data['accessibility']
data['neighborhood_zoning'] = data['neighborhood_quality'] * (data['zoning'] == 2)
data['log_land_size'] = np.log(data['land_size'])

# Prepare features and target
X = data[FEATURE_ORDER]  # Use explicit order
y = data['land_price']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with best parameters (simplified for brevity)
model = XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Save model, scaler, and feature order
joblib.dump(model, 'improved_land_price_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(FEATURE_ORDER, 'feature_order.pkl')