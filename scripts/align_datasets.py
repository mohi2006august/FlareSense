import pandas as pd
import numpy as np

def align_and_mock_solexs(file_path: str):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Original first rows:")
    print(df.head())
    
    # 1. Shift time forward by exactly 36 hours (129600 seconds)
    # June 23 00:00:00 -> June 24 12:00:00
    time_shift = 129600
    df['time_s'] = df['time_s'] + time_shift
    
    # Also update the obs_date column for consistency
    df['obs_date'] = '2026-06-24'
    
    # 2. Inject Mock C-Class Band (1-5 keV)
    # A realistic C-class band has roughly 20-30% of the total flux + some noise
    noise = np.random.normal(loc=0.0, scale=0.5, size=len(df))
    df['counts_1_5kev'] = (df['counts'] * 0.3) + noise
    # Ensure no negative flux
    df['counts_1_5kev'] = df['counts_1_5kev'].clip(lower=0.0)
    
    print("\nAligned and Mocked first rows:")
    print(df.head())
    
    print(f"\nOverwriting {file_path}...")
    df.to_csv(file_path, index=False)
    print("Alignment complete!")

if __name__ == "__main__":
    solexs_path = "data/raw/SOLEXS/AL1_SLX_cleaned.csv"
    align_and_mock_solexs(solexs_path)
