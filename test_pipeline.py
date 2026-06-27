import os
import torch
import numpy as np
import pandas as pd
from stage1.loader import FitsDataLoader
from stage1.pipeline import DataPreprocessor
from stage1.embeddings import PatchEmbedding
from stage1.sync import TimeSynchronizer
from astropy.time import Time

def test_pipeline():
    print("=== Testing Stage 1: Data Preprocessing Pipeline ===")
    
    # Initialize components
    loader = FitsDataLoader("data/raw")
    sync = TimeSynchronizer(solexs_epoch='2023-01-01T00:00:00')
    preprocessor = DataPreprocessor()
    embedder = PatchEmbedding() # Default params: 21600 seq_len, 30 patch_size, 256 dim
    
    # 1. Generate Mock Data with mismatched raw time formats
    target_len = 21600
    print("Generating Mock SoLEXS Data (seconds since epoch)...")
    # SoLEXS starts at 100 seconds past epoch, with 1-second cadence
    solexs_time = np.arange(100, 100 + target_len)
    solexs_flux = np.random.normal(loc=50.0, scale=5.0, size=target_len)
    df_solexs = pd.DataFrame({'TIME': solexs_time, 'FLUX': solexs_flux})
    
    print("Generating Mock HEL1OS Data (MJD)...")
    # HEL1OS starts precisely at the same absolute time as SoLEXS for this test
    # 2023-01-01T00:00:00 is MJD 59945.0
    # Add 100 seconds to match SoLEXS start
    start_mjd = 59945.0 + (100.0 / 86400.0)
    helios_time = start_mjd + (np.arange(target_len) / 86400.0)
    helios_flux = np.random.normal(loc=100.0, scale=10.0, size=target_len)
    helios_error = np.ones(target_len)
    
    # Introduce some artificial spikes and gaps in HEL1OS
    helios_flux[500] = 10000.0
    helios_flux[10200] = 5000.0
    helios_flux[15000:15120] = np.nan # 2-minute gap
    
    df_helios = pd.DataFrame({'TIME': helios_time, 'FLUX': helios_flux, 'ERROR': helios_error})
    
    try:
        # 2. Time Synchronization
        print("\nRunning TimeSynchronizer to align SoLEXS and HEL1OS onto 1-second UTC grid...")
        master_df = sync.sync(df_solexs, df_helios)
        print(f"Synchronized Master DataFrame Shape: {master_df.shape}")
        print("First 3 rows of Synchronized Data:")
        print(master_df.head(3))
        
        # 3. Preprocess Data (Running on HEL1OS channel as an example)
        print("\nRunning Preprocessor on HEL1OS channel (Spike removal, Gap fill, Log10, Z-score)...")
        # Preprocessor expects a 'FLUX' column, but master_df has 'HELIOS_FLUX' and 'SOLEXS_FLUX'
        # We rename it temporarily for the preprocessor, or update process_channel to accept column names.
        # process_channel accepts flux_column argument.
        processed_df = preprocessor.process_channel(master_df, flux_column='HELIOS_FLUX', is_training=True)
        print(f"Preprocessed DataFrame Shape: {processed_df.shape}")
        
        # We need exactly 21600 steps (6 hours at 1-sec cadence) for the embedder
        flux_array = processed_df['HELIOS_FLUX'].values
        mask_array = processed_df['is_gap'].values
        
        if len(flux_array) > target_len:
            flux_array = flux_array[:target_len]
            mask_array = mask_array[:target_len]
        elif len(flux_array) < target_len:
            pad_len = target_len - len(flux_array)
            flux_array = np.pad(flux_array, (0, pad_len), 'constant', constant_values=0)
            mask_array = np.pad(mask_array, (0, pad_len), 'constant', constant_values=True)
            
        print(f"Padded/Truncated array to sequence length: {target_len}")
            
        # 4. Patch Embedding
        x_tensor = torch.tensor(flux_array, dtype=torch.float32).view(1, 1, -1)
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool).view(1, -1)
        
        print(f"\nPassing into PatchEmbedding layer: Input Shape {x_tensor.shape}")
        embeddings = embedder(x_tensor, mask_tensor)
        
        print("\n=== Test Successful! ===")
        print(f"Final Mamba Token Embeddings Shape: {embeddings.shape}")
        
        if tuple(embeddings.shape) == (1, 720, 256):
            print("=> SHAPE VERIFIED: PERFECT MATCH!")
        else:
            print("=> SHAPE MISMATCH!")
            
    except Exception as e:
        print(f"Test Failed: {e}")
        
if __name__ == "__main__":
    test_pipeline()
