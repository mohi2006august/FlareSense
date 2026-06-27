import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple

from stage1.pipeline import DataPreprocessor
from stage1.labels import FlareLabelGenerator
from stage1.fallback import DataPatcher

class PBCATDataset(Dataset):
    """
    PyTorch Dataset for PBCAT-M.
    Slides a 6-hour window (21,600 steps) across the synchronized Aditya-L1 data.
    """
    
    def __init__(
        self, 
        synchronized_df: pd.DataFrame, 
        label_generator: FlareLabelGenerator,
        goes_fallback_df: pd.DataFrame = None,
        window_size_sec: int = 21600,  # 6 hours
        stride_sec: int = 1800         # 30 minutes
    ):
        """
        Args:
            synchronized_df: The DataFrame output by TimeSynchronizer 
            label_generator: Instance of FlareLabelGenerator for ground truths
            goes_fallback_df: Supplementary GOES DataFrame for gap patching
            window_size_sec: Size of each sequence in seconds
            stride_sec: How far to slide the window forward for each sample
        """
        self.df = synchronized_df.sort_values('UTC_TIME').reset_index(drop=True)
        
        if goes_fallback_df is not None:
            print("Running DataPatcher to fill large gaps with GOES data...")
            patcher = DataPatcher(gap_threshold_sec=300)
            self.df = patcher.patch_channel(self.df, goes_fallback_df, primary_col='SOLEXS_FLUX', fallback_col='GOES_FLUX')
            if 'SOLEXS_FLUX_1_5KEV' in self.df.columns:
                self.df = patcher.patch_channel(self.df, goes_fallback_df, primary_col='SOLEXS_FLUX_1_5KEV', fallback_col='GOES_FLUX')
            self.df = patcher.patch_channel(self.df, goes_fallback_df, primary_col='HELIOS_FLUX', fallback_col='GOES_FLUX')
        self.label_gen = label_generator
        self.window_size = window_size_sec
        self.stride = stride_sec
        
        # We need independent preprocessors for each channel because their statistical 
        # means and standard deviations (for z-scoring) are completely different.
        self.solexs_preprocessor = DataPreprocessor()
        self.solexs_cclass_preprocessor = DataPreprocessor()
        self.helios_preprocessor = DataPreprocessor()
        
        # Pre-fit the preprocessors on the entire dataset to get global mean/std
        # This solves the issue of "stats computed on mock/batch"
        print("Fitting global Z-score statistics on the full dataset...")
        self.solexs_preprocessor.fit_transform_zscore(self.df['SOLEXS_FLUX'].dropna())
        
        if 'SOLEXS_FLUX_1_5KEV' in self.df.columns:
            self.solexs_cclass_preprocessor.fit_transform_zscore(self.df['SOLEXS_FLUX_1_5KEV'].dropna())
            
        self.helios_preprocessor.fit_transform_zscore(self.df['HELIOS_FLUX'].dropna())
        
        # Calculate how many full sliding windows we can fit
        total_seconds = len(self.df)
        self.num_windows = max(0, (total_seconds - self.window_size) // self.stride + 1)
        
    def __len__(self):
        return self.num_windows
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts the chunk, preprocesses it, and returns the final tensors.
        Returns:
            solexs_tensor: (1, 21600)
            solexs_mask:   (21600,)
            helios_tensor: (1, 21600)
            helios_mask:   (21600,)
            labels:        [class_label, onset_hours]
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Extract the 6-hour chunk
        chunk = self.df.iloc[start_idx:end_idx].copy()
        
        window_start = chunk['UTC_TIME'].iloc[0]
        window_end = chunk['UTC_TIME'].iloc[-1]
        
        # 1. Preprocess SoLEXS Total
        solexs_processed = self.solexs_preprocessor.process_channel(chunk, flux_column='SOLEXS_FLUX', is_training=False)
        solexs_flux = solexs_processed['SOLEXS_FLUX'].values
        solexs_mask = solexs_processed['is_gap'].values
        
        # 2. Preprocess SoLEXS C-Class (1-5 keV)
        if 'SOLEXS_FLUX_1_5KEV' in chunk.columns:
            cclass_processed = self.solexs_cclass_preprocessor.process_channel(chunk, flux_column='SOLEXS_FLUX_1_5KEV', is_training=False)
            cclass_flux = cclass_processed['SOLEXS_FLUX_1_5KEV'].values
            cclass_mask = cclass_processed['is_gap'].values
        else:
            cclass_flux = solexs_flux
            cclass_mask = solexs_mask
        
        # 3. Preprocess HEL1OS
        helios_processed = self.helios_preprocessor.process_channel(chunk, flux_column='HELIOS_FLUX', is_training=False)
        helios_flux = helios_processed['HELIOS_FLUX'].values
        helios_mask = helios_processed['is_gap'].values
        
        # 4. Generate Label for this specific 6-hour window
        flare_class, onset_time = self.label_gen.get_label_for_window(window_start, window_end)
        
        # 5. Convert to PyTorch Tensors
        # Add channel dimension (1, 21600)
        s_tensor = torch.tensor(solexs_flux, dtype=torch.float32).unsqueeze(0)
        s_mask = torch.tensor(solexs_mask, dtype=torch.bool)
        
        c_tensor = torch.tensor(cclass_flux, dtype=torch.float32).unsqueeze(0)
        c_mask = torch.tensor(cclass_mask, dtype=torch.bool)
        
        h_tensor = torch.tensor(helios_flux, dtype=torch.float32).unsqueeze(0)
        h_mask = torch.tensor(helios_mask, dtype=torch.bool)
        
        label_tensor = torch.tensor([flare_class, onset_time], dtype=torch.float32)
        
        return s_tensor, s_mask, c_tensor, c_mask, h_tensor, h_mask, label_tensor
