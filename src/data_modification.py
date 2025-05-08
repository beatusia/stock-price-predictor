# src/data_modification.py
# This script processes stock data CSV files to remove unwanted rows and add a ticker column.
import pandas as pd
import os

# Define the directory where your stock files are located
directory = 'path_to_your_directory'  # Change this to your directory path

# List all files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.csv')]  # Filter for CSV files

# Loop through each file
for file in files:
    # Construct the full file path
    file_path = os.path.join(directory, file)
    
    # Load the stock data
    df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is the index (Date)
    
    # Check the first few rows of the DataFrame to debug (remove later)
    print(f"First few rows of {file}:")
    print(df.head())

    # Remove the first two rows using iloc, but keep the index intact
    df = df.iloc[2:]  # Keep the Date column intact and drop first two rows
    
    # Add the ticker column with the filename (remove the file extension)
    ticker_name = os.path.splitext(file)[0]  # Extracts ticker name from filename
    df['TICKER'] = ticker_name
    
    # Save the modified DataFrame back to a new file (optional)
    new_file_path = os.path.join(directory, f"modified_{file}")
    df.to_csv(new_file_path, index=True)  # Save with the 'Date' index intact
    
    # Print the modified DataFrame (for checking)
    print(f"Processed {file}")
    print(df.head())  # Print the first few rows to verify

# Optionally, you can remove the original file if needed
# os.remove(file_path)  # Uncomment to delete the original file after processing
# Note: Make sure to replace 'path_to_your_directory' with the actual path to your directory.

# The code below is for renaming the index and reordering the columns in the modified CSV files
# Define the directory path
directory = '/data/raw/modified'  # Update this if needed

# Loop through each CSV file in the directory
for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_path = os.path.join(directory, file)
        
        # Read the CSV using the first column as the index
        df = pd.read_csv(file_path, index_col=0)
        
        # Convert the index (which is 'Price' mislabeled) into a new column called 'Date'
        df.index.name = 'Date'  # Set proper name for the index
        df.reset_index(inplace=True)

        # Ensure 'TICKER' exists (skip or warn if not)
        if 'TICKER' not in df.columns:
            print(f"Warning: 'TICKER' column not found in {file}")
            continue
        
        # Define the desired column order
        desired_order = ['Date', 'TICKER', 'Close', 'High', 'Low', 'Open', 'Volume']
        # Keep only columns that exist in the current DataFrame
        ordered_columns = [col for col in desired_order if col in df.columns]

        # Reorder the DataFrame
        df = df[ordered_columns]
        
        # Save back to the same file
        df.to_csv(file_path, index=False)
