import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load Data
# Replace 'land_data.csv' with your actual dataset file
# Ensure the dataset contains relevant features and the target variable 'land_price'
data = pd.read_csv("synthetic_land_data.csv")

# Check for missing values and fill them (basic handling; customize as needed)
data.fillna(data.mean(), inplace=True)

# Separate features (X) and target (y)
X = data.drop(columns=["land_price"])  # Features
y = data["land_price"]  # Target variable

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train XGBoost Regressor
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

# Step 5: Feature Importance
feature_importance = model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]

# Print feature importance scores
print("\nFeature Importance Scores:")
for i in sorted_indices:
    print(f"{X.columns[i]}: {feature_importance[i]:.4f}")

# Step 6: Visualization of Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_indices], feature_importance[sorted_indices], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Land Price Prediction")
plt.gca().invert_yaxis()  # Invert Y-axis for better visualization
plt.show()

# Step 7: Save the Model (Optional)
# Save the trained model for future use
import joblib
joblib.dump(model, "land_price_prediction_model.pkl")

# Step 8: Make Predictions for New Data
# Replace this example input with actual data for prediction
new_data = pd.DataFrame({
    "location": [3],  # Example encoded value
    "accessibility": [5],
    "neighborhood_quality": [4],
    "zoning": [2],
    "historical_trends": [300],
    "land_size": [4500]
})
new_price_prediction = model.predict(new_data)
print(f"Predicted Land Price: {new_price_prediction[0]:.2f}")