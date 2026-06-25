import pandas as pd
import numpy as np

class DataPreprocessor:
    """
    Handles spike removal, gap filling, log10 transform, and z-score normalization
    for Aditya-L1 1-second cadence time-series data.
    """
    def __init__(self, spike_window: int = 5, gap_fill_limit: int = 60, spike_threshold: float = 3.0):
        self.spike_window = spike_window
        self.gap_fill_limit = gap_fill_limit
        self.spike_threshold = spike_threshold
        
        self.mean = None
        self.std = None

    def remove_spikes(self, series: pd.Series) -> pd.Series:
        """
        Rolling median filter to detect and replace unphysical spikes.
        """
        rolling_median = series.rolling(window=self.spike_window, center=True).median()
        # Calculate deviation from median
        mad = (series - rolling_median).abs()
        rolling_mad = mad.rolling(window=self.spike_window, center=True).median()
        
        # Flag spikes (where deviation is greater than threshold * rolling MAD)
        # Add a small epsilon to avoid division by zero
        is_spike = mad > (self.spike_threshold * rolling_mad + 1e-8)
        
        # Replace spikes with median
        cleaned_series = series.copy()
        cleaned_series[is_spike] = rolling_median[is_spike]
        return cleaned_series

    def fill_gaps(self, series: pd.Series) -> pd.Series:
        """
        Forward-fill for gaps < 60s. Longer gaps are left as NaN 
        to be handled by the gap-aware masking token.
        """
        # Forward fill up to gap_fill_limit
        return series.ffill(limit=self.gap_fill_limit)

    def log10_transform(self, series: pd.Series) -> pd.Series:
        """
        Compresses 4-5 orders of flux magnitude to a uniform scale.
        """
        # Add small epsilon to handle zeros/negative values from noise
        return np.log10(series.clip(lower=1e-8))

    def fit_transform_zscore(self, series: pd.Series) -> pd.Series:
        """
        Zero mean, unit variance from training statistics.
        Calculates mean and std, then transforms.
        """
        self.mean = series.mean()
        self.std = series.std()
        
        if self.std == 0 or np.isnan(self.std):
            self.std = 1.0 # Prevent division by zero
            
        return (series - self.mean) / self.std
        
    def transform_zscore(self, series: pd.Series) -> pd.Series:
        """
        Apply already calculated mean and std for z-score normalization.
        """
        if self.mean is None or self.std is None:
            raise ValueError("DataPreprocessor must be fitted before calling transform_zscore")
        return (series - self.mean) / self.std

    def process_channel(self, df: pd.DataFrame, flux_column: str = 'FLUX', is_training: bool = True) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline on a single channel DataFrame.
        """
        processed_df = df.copy()
        
        # 1. Spike Removal
        processed_df[flux_column] = self.remove_spikes(processed_df[flux_column])
        
        # 2. Gap Filling
        processed_df[flux_column] = self.fill_gaps(processed_df[flux_column])
        
        # 3. Log10 Transform
        processed_df[flux_column] = self.log10_transform(processed_df[flux_column])
        
        # 4. Z-Score Normalization
        if is_training:
            processed_df[flux_column] = self.fit_transform_zscore(processed_df[flux_column])
        else:
            processed_df[flux_column] = self.transform_zscore(processed_df[flux_column])
            
        # Any remaining NaNs (gaps > 60s) will be replaced with 0.0 (which is the mean after z-score)
        # The PatchEmbedding layer will use a mask to replace these with a learnable token
        # For PyTorch compatibility, we fill NaN with 0 and create a separate mask boolean array
        processed_df['is_gap'] = processed_df[flux_column].isna()
        processed_df[flux_column] = processed_df[flux_column].fillna(0.0)
        
        return processed_df
