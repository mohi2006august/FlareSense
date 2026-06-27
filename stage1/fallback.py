import pandas as pd
import numpy as np

class DataPatcher:
    """
    Multi-Source Resilience Layer.
    Patches large data gaps (e.g. from satellite eclipses) in Aditya-L1 data
    using a secondary source (e.g. GOES-R or STEREO).
    """
    
    def __init__(self, gap_threshold_sec: int = 300):
        """
        Args:
            gap_threshold_sec: Only gaps larger than this (e.g. 5 mins) will be patched.
                               Smaller gaps are handled by linear interpolation in the Preprocessor.
        """
        self.gap_threshold = gap_threshold_sec
        
    def patch_channel(self, master_df: pd.DataFrame, fallback_df: pd.DataFrame, 
                      primary_col: str, fallback_col: str = 'FLUX') -> pd.DataFrame:
        """
        Finds large NaNs in `primary_col` and dynamically scales and inserts 
        the data from `fallback_df` into those gaps.
        
        Args:
            master_df: The synchronized 1-sec UTC timeline dataframe.
            fallback_df: The GOES/STEREO dataframe (must have 'UTC_TIME' and fallback_col).
            primary_col: The column in master_df to patch (e.g., 'SOLEXS_FLUX')
            fallback_col: The flux column in fallback_df.
            
        Returns:
            The patched master_df.
        """
        df = master_df.copy()
        
        # Ensure fallback data is properly indexed by UTC_TIME
        if 'UTC_TIME' in fallback_df.columns:
            fallback = fallback_df.set_index('UTC_TIME')
        else:
            fallback = fallback_df.copy()
            
        # Reindex fallback to precisely match the master 1-second grid
        # We forward fill up to 3 seconds for GOES data if it's slightly off-cadence
        fallback = fallback.reindex(df['UTC_TIME'], method='ffill', limit=3)
        
        # Identify missing data in the primary channel
        is_missing = df[primary_col].isna()
        
        # We need to find contiguous blocks of True
        # Using a cumulative sum to create block IDs
        block_ids = (~is_missing).cumsum()
        
        # Group by block IDs and find gaps
        for block_id, group in df[is_missing].groupby(block_ids):
            gap_length = len(group)
            
            if gap_length > self.gap_threshold:
                gap_start_idx = group.index[0]
                gap_end_idx = group.index[-1]
                
                # EDGE-MATCHING CROSS-CALIBRATION
                # We need the primary value right before the gap, and the fallback value at that exact same time
                if gap_start_idx > 0:
                    edge_idx = gap_start_idx - 1
                    primary_edge = df.loc[edge_idx, primary_col]
                    fallback_edge = fallback.iloc[edge_idx][fallback_col]
                    
                    if not np.isnan(primary_edge) and not np.isnan(fallback_edge) and fallback_edge > 0:
                        # Scaling ratio = Primary / Fallback
                        scale_factor = primary_edge / fallback_edge
                        
                        # Extract the fallback data for the gap duration, scale it, and insert
                        fallback_segment = fallback.iloc[gap_start_idx:gap_end_idx+1][fallback_col]
                        patched_segment = fallback_segment * scale_factor
                        
                        df.loc[gap_start_idx:gap_end_idx, primary_col] = patched_segment.values
                        
                        # Add a flag to indicate this data is synthetic/fallback
                        # The preprocessor uses 'is_gap' (boolean). 
                        # We will set a new column 'is_fallback' so the network knows the source changed.
                        if 'is_fallback' not in df.columns:
                            df['is_fallback'] = 0
                        df.loc[gap_start_idx:gap_end_idx, 'is_fallback'] = 1
                        
                        print(f"Patched a {gap_length}-sec gap starting at {df.loc[gap_start_idx, 'UTC_TIME']} using scale factor {scale_factor:.4f}")
                    else:
                        print(f"Could not patch gap starting at {df.loc[gap_start_idx, 'UTC_TIME']} - missing edge calibration data.")
                        
        return df
