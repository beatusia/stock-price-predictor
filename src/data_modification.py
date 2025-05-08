# If you find that your csv files have two empty rows with the ticker name in the first row, use this script to remove them
import pandas as pd
import os

# Define the directory where your stock files are located
directory = 'path_to_your_directory'  # Change this to your directory path

# List all files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.csv')]  # Adjust extension if needed

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
    
    # Optionally, reset the index if necessary (so the Date column becomes a regular column)
    # df = df.reset_index(drop=False)  # Uncomment if you want the 'Date' column to become a regular column
    
    # Save the modified DataFrame back to a new file (optional)
    new_file_path = os.path.join(directory, f"modified_{file}")
    df.to_csv(new_file_path, index=True)  # Save with the 'Date' index intact
    
    # Print the modified DataFrame (for checking)
    print(f"Processed {file}")
    print(df.head())  # Print the first few rows to verify

# Optionally, you can remove the original file if needed
# os.remove(file_path)  # Uncomment to delete the original file after processing

# Note: Make sure to replace 'path_to_your_directory' with the actual path to your directory.