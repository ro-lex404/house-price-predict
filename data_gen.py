import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_land_data(num_samples=100000):
    np.random.seed(42)
    
    # Generate features with realistic distributions
    location = np.random.beta(2, 2, num_samples) * 10
    accessibility = np.clip(np.random.normal(5, 2, num_samples), 0, 10)
    neighborhood_quality = np.clip(np.random.beta(4, 2, num_samples) * 10, 0, 10)
    zoning = np.random.choice([1, 2, 3], size=num_samples, p=[0.7, 0.2, 0.1])
    historical_trends = np.clip(np.random.lognormal(5.5, 0.4, num_samples), 200, 1000)
    land_size = np.clip(np.random.lognormal(8.5, 0.6, num_samples), 1000, 50000)
    
    # Price calculation with interaction effects
    base_price = (
        np.exp(location / 5) * 5000 +
        accessibility ** 2 * 1000 +
        neighborhood_quality ** 1.5 * 2000 +
        (zoning == 2) * 50000 +
        (zoning == 3) * (-20000) +
        historical_trends * 3 +
        np.sqrt(land_size) * 100
    )
    
    interaction_effects = (
        (location * accessibility) * 500 +
        (neighborhood_quality * (zoning == 2)) * 1000
    )
    
    noise = np.random.normal(0, 0.1, num_samples) * base_price
    land_price = base_price + interaction_effects + noise
    
    data = pd.DataFrame({
        "location": location,
        "accessibility": accessibility,
        "neighborhood_quality": neighborhood_quality,
        "zoning": zoning,
        "historical_trends": historical_trends,
        "land_size": land_size,
        "land_price": land_price
    })
    
    return data

# Generate data and save to CSV
data = generate_land_data()
data.to_csv("synthetic_land_data.csv", index=False)
print(f"Generated {len(data)} records and saved to 'synthetic_land_data.csv'")

# Quick data validation
print("\nData Summary:")
print(data.describe())