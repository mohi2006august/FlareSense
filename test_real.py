import os
import numpy as np
import pandas as pd
from stage1.loader import FitsDataLoader
from stage1.sync import TimeSynchronizer
from astropy.io import fits

def main():
    print("Testing the pipeline on REAL datasets")
    print("-" * 50)
    
    loader = FitsDataLoader("data/raw")
    
    try:
        # Load HEL1OS CSV
        helios_file = "HEL1OS/HLS_cleaned.csv"
        df_helios = loader.load_helios(helios_file)
        
        # Load SoLEXS CSV
        solexs_file = "SOLEXS/AL1_SLX_cleaned.csv"
        df_solexs = loader.load_solexs(solexs_file)
        
        print("SoLEXS DataFrame:")
        print(df_solexs.head())
        
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return

    from stage1.pipeline import DataPreprocessor
    
    print("\nRunning Time Synchronizer...")
    sync = TimeSynchronizer(solexs_epoch='2023-01-01T00:00:00')
    try:
        master_df = sync.sync(df_solexs, df_helios)
        print("\nSynchronized Master DataFrame (First 5 rows with data):")
        print(master_df.dropna().head(5))
        
        print("\n=== Actual Raw FLUX values === ")
        print("SoLEXS Mean Flux:", master_df['SOLEXS_FLUX'].mean())
        print("HEL1OS Mean Flux:", master_df['HELIOS_FLUX'].mean())
        
        # Now CLEAN the dataset
        print("\n=== Cleaning Dataset ===")
        preprocessor = DataPreprocessor()
        
        # Fit and Transform SoLEXS
        preprocessor.fit_transform_zscore(master_df['SOLEXS_FLUX'].dropna())
        cleaned_solexs = preprocessor.process_channel(master_df, flux_column='SOLEXS_FLUX', is_training=False)
        
        # Fit and Transform HEL1OS
        preprocessor.fit_transform_zscore(master_df['HELIOS_FLUX'].dropna())
        cleaned_helios = preprocessor.process_channel(master_df, flux_column='HELIOS_FLUX', is_training=False)
        
        # Combine the cleaned columns back together
        master_df['SOLEXS_FLUX_CLEAN'] = cleaned_solexs['SOLEXS_FLUX']
        master_df['SOLEXS_GAP'] = cleaned_solexs['is_gap']
        
        master_df['HELIOS_FLUX_CLEAN'] = cleaned_helios['HELIOS_FLUX']
        master_df['HELIOS_GAP'] = cleaned_helios['is_gap']
        
        print("\nCleaned Data (First 5 rows):")
        print(master_df[['UTC_TIME', 'SOLEXS_FLUX_CLEAN', 'HELIOS_FLUX_CLEAN']].head())
        
        output_csv = "data/cleaned_real_dataset.csv"
        master_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully cleaned and saved the dataset to: {output_csv}")
        
    except Exception as e:
        print(f"Synchronization/Cleaning failed: {e}")

if __name__ == "__main__":
    main()
