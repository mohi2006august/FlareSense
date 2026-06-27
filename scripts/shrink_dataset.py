import pandas as pd
import os

def shrink_csv(file_path: str, target_size_mb: float = 95.0):
    """
    Reads a massive CSV and drops rows from the end until the file size
    is comfortably below the target_size_mb limit (so it can be pushed to GitHub).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    current_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Initial file size of {file_path}: {current_size_mb:.2f} MB")
    
    if current_size_mb <= target_size_mb:
        print("File is already below the target size limit. No action needed.")
        return
        
    # Read the full dataset
    print("Loading CSV into memory...")
    df = pd.read_csv(file_path)
    initial_rows = len(df)
    
    # Calculate rough ratio of bytes per row
    bytes_per_row = (current_size_mb * 1024 * 1024) / initial_rows
    target_bytes = target_size_mb * 1024 * 1024
    
    # Estimate how many rows we should keep
    rows_to_keep = int(target_bytes / bytes_per_row)
    
    # Leave a safety margin of 2% just in case
    rows_to_keep = int(rows_to_keep * 0.98)
    
    print(f"Total rows: {initial_rows}. Shrinking down to roughly {rows_to_keep} rows...")
    df_shrunk = df.iloc[:rows_to_keep]
    
    print(f"Overwriting {file_path}...")
    df_shrunk.to_csv(file_path, index=False)
    
    new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Done! New file size: {new_size_mb:.2f} MB")

if __name__ == "__main__":
    helios_path = "data/raw/HEL1OS/HLS_cleaned.csv"
    shrink_csv(helios_path, target_size_mb=95.0)
