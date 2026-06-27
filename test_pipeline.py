import os
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from stage1.loader import FitsDataLoader
from stage1.sync import TimeSynchronizer
from stage1.labels import FlareLabelGenerator
from stage1.dataset import PBCATDataset
from stage1.embeddings import PatchEmbedding
from astropy.io import fits

def test_pipeline():
    print("=== Testing Stage 1: PyTorch Dataset & Label Generator ===")
    
    loader = FitsDataLoader("data/raw")
    
    # 1. Load Data
    helios_file = "HEL1OS/sdte/lightcurve_cdte1.fits"
    print(f"Loading {helios_file}...")
    df_helios = loader.load_helios(helios_file)
    df_helios = pd.DataFrame({col: df_helios[col].values.astype(np.float64) for col in df_helios.columns})

    solexs_file = "SOLEXS/SDD2/dataset3.pi"
    print(f"Loading {solexs_file}...")
    df_solexs = loader.load_solexs(solexs_file)
    df_solexs = pd.DataFrame({col: df_solexs[col].values.astype(np.float64) for col in df_solexs.columns})

    # 2. Time Synchronization
    print("\nRunning Time Synchronizer...")
    sync = TimeSynchronizer(solexs_epoch='2023-01-01T00:00:00')
    master_df = sync.sync(df_solexs, df_helios)
    print(f"Synchronized Master DataFrame Shape: {master_df.shape}")
    
    # 3. Create Mock GOES Fallback Data (CSV)
    print("\nGenerating Mock GOES Fallback Data...")
    goes_time = pd.date_range(start=master_df['UTC_TIME'].min(), end=master_df['UTC_TIME'].max(), freq='1s')
    goes_flux = np.random.normal(loc=1.0, scale=0.1, size=len(goes_time))
    # GOES is typically in Watts/m^2 (e.g. 1e-7), but we just use arbitrary numbers to test scaling
    mock_goes = pd.DataFrame({'time_tag': goes_time, 'flux': goes_flux})
    goes_path = "data/raw/mock_goes_flux.csv"
    mock_goes.to_csv(goes_path, index=False)
    
    # Load GOES via loader
    df_goes = loader.load_goes("mock_goes_flux.csv")
    
    # INJECT A GAP IN Aditya-L1 DATA TO TEST FALLBACK
    print("\nInjecting a 1-hour eclipse gap in Aditya-L1 master timeline...")
    gap_start = 5000
    gap_end = 5000 + 3600 # 1 hour
    master_df.loc[gap_start:gap_end, 'SOLEXS_FLUX'] = np.nan
    master_df.loc[gap_start:gap_end, 'SOLEXS_FLUX_1_5KEV'] = np.nan
    master_df.loc[gap_start:gap_end, 'HELIOS_FLUX'] = np.nan
    
    # 4. Create Mock Flare Catalog (CSV)
    # We will simulate a M-class flare in our window
    print("\nGenerating Mock GOES Flare Catalog...")
    mock_catalog = pd.DataFrame({
        'start_time': ['2023-01-01 02:00:00', '2023-01-01 14:00:00'],
        'peak_time':  ['2023-01-01 02:15:00', '2023-01-01 14:30:00'],
        'end_time':   ['2023-01-01 03:00:00', '2023-01-01 15:00:00'],
        'class':      ['M2.1', 'C1.5']
    })
    catalog_path = "data/raw/mock_goes_catalog.csv"
    mock_catalog.to_csv(catalog_path, index=False)
    
    label_gen = FlareLabelGenerator(catalog_path)
    print("Label Generator initialized successfully.")
    
    # 6. Initialize PyTorch Dataset
    print("\nInitializing PBCATDataset...")
    # We have exactly 24 hours of mock data.
    # 6-hour window, 30-min stride -> ~37 windows
    dataset = PBCATDataset(master_df, label_gen, goes_fallback_df=df_goes, window_size_sec=21600, stride_sec=1800)
    print(f"Dataset created with {len(dataset)} sliding windows.")
    
    # 5. Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 6. Fetch a Batch
    print("\nFetching first batch from DataLoader...")
    batch = next(iter(dataloader))
    s_tensor, s_mask, c_tensor, c_mask, h_tensor, h_mask, labels = batch
    
    print("\n=== Batch Shapes ===")
    print(f"SoLEXS Total Tensor: {s_tensor.shape} (Expected: [4, 1, 21600])")
    print(f"SoLEXS 1-5keV Tensor: {c_tensor.shape} (Expected: [4, 1, 21600])")
    print(f"SoLEXS Mask:   {s_mask.shape} (Expected: [4, 21600])")
    print(f"HEL1OS Tensor: {h_tensor.shape} (Expected: [4, 1, 21600])")
    print(f"Labels:        {labels.shape} (Expected: [4, 2])")
    
    print("\n=== Batch Labels (Class, Onset Hours) ===")
    print(labels)
    
    # 7. Test passing through Embedder (Stage 1 final step)
    print("\nTesting PatchEmbedding layer on SoLEXS channel...")
    embedder = PatchEmbedding()
    s_embeds = embedder(s_tensor, s_mask)
    print(f"Embedded SoLEXS Shape: {s_embeds.shape} (Expected: [4, 720, 256])")
    
    print("\n=== All Stage 1 Components Successfully Integrated! ===")

if __name__ == "__main__":
    test_pipeline()
