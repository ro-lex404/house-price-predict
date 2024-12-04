import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic features
num_samples = 1000
location = np.random.uniform(0, 10, num_samples)  # 0 = rural, 10 = city center
accessibility = np.random.uniform(0, 10, num_samples)  # 0 = poor access, 10 = excellent access
neighborhood_quality = np.random.uniform(0, 10, num_samples)  # 0 = low, 10 = high
zoning = np.random.choice([1, 2, 3], size=num_samples)  # 1 = residential, 2 = commercial, 3 = industrial
historical_trends = np.random.uniform(200, 500, num_samples)  # Past average price in $/sqft
land_size = np.random.uniform(1000, 10000, num_samples)  # Size in sqft

# Generate target variable (land price)
# Prices depend on weighted combination of features + random noise
land_price = (
    location * 1000 +
    accessibility * 500 +
    neighborhood_quality * 300 +
    (zoning == 2) * 10000 -  # Commercial zones add $10,000
    (zoning == 3) * 5000 +  # Industrial zones reduce $5,000
    historical_trends * 2 +
    land_size * 0.5 +
    np.random.normal(0, 10000, num_samples)  # Random noise
)

# Combine into a DataFrame
data = pd.DataFrame({
    "location": location,
    "accessibility": accessibility,
    "neighborhood_quality": neighborhood_quality,
    "zoning": zoning,
    "historical_trends": historical_trends,
    "land_size": land_size,
    "land_price": land_price
})

# Save to CSV
data.to_csv("synthetic_land_data.csv", index=False)
print("Synthetic dataset saved as 'synthetic_land_data.csv'")
