# src/data_modification.py
# This script processes stock data CSV files to remove unwanted rows and add a ticker column.
import pandas as pd
import os

# Define paths
TRUE_RAW_DIR = '/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/true_raw'
MODIFIED_DIR = '/Users/beatawyspianska/Desktop/AIML_Projects/predict_stock_price/stock-price-predictor/data/raw/modified'

# Process each CSV in true_raw
for file in os.listdir(TRUE_RAW_DIR):
    if file.endswith('.csv'):
        input_path = os.path.join(TRUE_RAW_DIR, file)

        # Read CSV and remove first two rows
        df = pd.read_csv(input_path, index_col=0)
        df = df.iloc[2:]

        # Add 'TICKER' column based on filename
        ticker = os.path.splitext(file)[0]
        df['TICKER'] = ticker

        # Move index (Date) to a column
        df.insert(0, 'Date', df.index)
        df.reset_index(drop=True, inplace=True)

        # Reorder columns (only include those that exist)
        desired_order = ['Date', 'TICKER', 'Close', 'High', 'Low', 'Open', 'Volume']
        final_columns = [col for col in desired_order if col in df.columns]
        df = df[final_columns]

        # Save cleaned file to the modified directory
        output_path = os.path.join(MODIFIED_DIR, file)
        df.to_csv(output_path, index=False)

        print(f"✅ Processed and saved: {output_path}")
