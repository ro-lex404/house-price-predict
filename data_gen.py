import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate base features
    location = np.random.uniform(0, 10, num_samples)
    accessibility = np.clip(location + np.random.normal(0, 1, num_samples), 0, 10)
    neighborhood_quality = np.clip(0.7 * location + 0.3 * accessibility + np.random.normal(0, 1, num_samples), 0, 10)
    
    # Generate zoning (typical Bangalore distribution)
    zoning_probs = [0.75, 0.20, 0.05]  # Residential, Commercial, Industrial
    zoning = np.random.choice([1, 2, 3], num_samples, p=zoning_probs)
    
    # Generate land sizes (in sq ft)
    land_size = np.exp(np.random.normal(7.5, 0.5, num_samples))  # Typical Bangalore plot sizes
    
    # Historical trends (based on Bangalore average rates)
    historical_trends = 4000 + 500 * location + 200 * (zoning == 2) * location
    historical_trends = historical_trends + np.random.normal(0, 200, num_samples)
    
    # Calculate price (₹ per sq ft)
    base_price = (
        1000 * location +  # Location premium (0-10,000)
        500 * accessibility +  # Accessibility premium (0-5,000)
        300 * neighborhood_quality +  # Quality premium (0-3,000)
        2000 * (zoning == 2) +  # Commercial premium
        -1000 * (zoning == 3) +  # Industrial discount
        0.2 * historical_trends  # Historical influence
    )
    
    # Add market variations
    price = base_price * (1 + np.random.normal(0, 0.15, num_samples))
    
    # Clip prices to realistic Bangalore ranges (₹ per sq ft)
    price = np.clip(price, 3500, 25000)
    
    df = pd.DataFrame({
        'location': location,
        'accessibility': accessibility,
        'neighborhood_quality': neighborhood_quality,
        'zoning': zoning,
        'historical_trends': historical_trends,
        'land_size': land_size,
        'land_price': price
    })
    
    return df

# Generate data and train model
df = generate_synthetic_data(10000)
df.to_csv('land_price_data.csv', index=False)
