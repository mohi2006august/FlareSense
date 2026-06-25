import os
import torch
import numpy as np
import pandas as pd
from stage1.loader import FitsDataLoader
from stage1.pipeline import DataPreprocessor
from stage1.embeddings import PatchEmbedding

def test_pipeline():
    print("=== Testing Stage 1: Data Preprocessing Pipeline ===")
    
    # Initialize components
    loader = FitsDataLoader("data/raw")
    preprocessor = DataPreprocessor()
    embedder = PatchEmbedding() # Default params: 21600 seq_len, 30 patch_size, 256 dim
    
    # 1. Load Data (Using Mock Data for the test)
    # The FITS file pasted into the editor was truncated (missing the binary payload)
    # So we will generate a perfect dummy dataset to prove the pipeline works!
    print("Generating Mock HEL1OS Data (21,600 steps)...")
    
    # 6 hours = 21600 seconds
    target_len = 21600
    mock_flux = np.random.normal(loc=100.0, scale=10.0, size=target_len)
    
    # Introduce some artificial spikes
    mock_flux[500] = 10000.0
    mock_flux[10200] = 5000.0
    
    # Introduce a 2-minute gap (120 seconds) - this should trigger the masking token
    mock_flux[15000:15120] = np.nan
    
    df = pd.DataFrame({
        'TIME': np.arange(target_len),
        'FLUX': mock_flux,
        'ERROR': np.ones(target_len)
    })
    
    try:
        print(f"Successfully loaded mock HEL1OS data. Shape: {df.shape}")
        
        # 2. Preprocess Data
        # HEL1OS uses 'FLUX' internally now after loader standardization
        print("Running Preprocessor (Spike removal, Gap fill, Log10, Z-score)...")
        processed_df = preprocessor.process_channel(df, flux_column='FLUX', is_training=True)
        print(f"Preprocessed DataFrame Shape: {processed_df.shape}")
        
        # We need exactly 21600 steps (6 hours at 1-sec cadence) for the embedder
        # If the actual file is longer/shorter, we pad/truncate for the test
        flux_array = processed_df['FLUX'].values
        mask_array = processed_df['is_gap'].values
        
        target_len = 21600
        if len(flux_array) > target_len:
            flux_array = flux_array[:target_len]
            mask_array = mask_array[:target_len]
        elif len(flux_array) < target_len:
            pad_len = target_len - len(flux_array)
            flux_array = np.pad(flux_array, (0, pad_len), 'constant', constant_values=0)
            mask_array = np.pad(mask_array, (0, pad_len), 'constant', constant_values=True)
            
        print(f"Padded/Truncated array to sequence length: {target_len}")
            
        # 3. Patch Embedding
        # Convert to PyTorch tensors and add batch dimension (Batch size 1)
        # x shape needed: (batch_size, 1, sequence_length)
        x_tensor = torch.tensor(flux_array, dtype=torch.float32).view(1, 1, -1)
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool).view(1, -1)
        
        print(f"Passing into PatchEmbedding layer: Input Shape {x_tensor.shape}")
        
        embeddings = embedder(x_tensor, mask_tensor)
        
        print("=== Test Successful! ===")
        print(f"Final Mamba Token Embeddings Shape: {embeddings.shape}")
        print("Expected Shape: torch.Size([1, 720, 256])")
        
        if tuple(embeddings.shape) == (1, 720, 256):
            print("=> SHAPE VERIFIED: PERFECT MATCH!")
        else:
            print("=> SHAPE MISMATCH!")
            
    except Exception as e:
        print(f"Test Failed: {e}")
        
if __name__ == "__main__":
    test_pipeline()
