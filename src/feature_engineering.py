# src/feature_engineering.py
# This script creates High-Low, Price-Open and lagged features for closed prices for the past 10 trading days
import os
import pandas as pd

# Set the directory containing your stock CSVs
directory = '/data/raw/modified' # Replace with your actual path

# Loop through all CSV files in the folder
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        
        # Load the CSV
        data = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_cols = ['High', 'Low', 'Close', 'Open']
        if not all(col in data.columns for col in required_cols):
            print(f"Skipping {file}: missing required columns.")
            continue

        # Add engineered features
        data['High-Low'] = data['High'] - data['Low']
        data['Price-Open'] = data['Close'] - data['Open']
        
        # Create lag features for 'Close'
        for i in range(1, 11):
            data[f'Close_lag{i}'] = data['Close'].shift(i)
        
        # Drop rows with any NaNs caused by lagging
        data = data.dropna(subset=[f'Close_lag{n}' for n in range(1, 11)]).reset_index(drop=True)

        # Save the modified file back
        data.to_csv(file_path, index=False)

        print(f"Processed and saved: {file}")
