import pandas as pd
import numpy as np
from astropy.time import Time

class TimeSynchronizer:
    """
    Synchronizes SoLEXS and HEL1OS data onto a unified 1-second grid.
    Converts raw timestamps (TIME, MJD) to absolute UTC datetimes.
    """
    def __init__(self, solexs_epoch: str = '2023-01-01T00:00:00'):
        # Configurable epoch for SoLEXS in case the ISRO standard changes or is different
        self.solexs_epoch = Time(solexs_epoch, format='isot', scale='utc')
        
    def sync(self, df_solexs: pd.DataFrame, df_helios: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw SoLEXS and HEL1OS dataframes, standardizes their time formats,
        and merges them onto a master 1-second cadence UTC timeline.
        
        Expected SoLEXS columns: ['TIME', 'FLUX']
        Expected HEL1OS columns: ['TIME', 'FLUX', 'ERROR']  (Loader standardized MJD to TIME)
        """
        # 1. Standardize SoLEXS time (assuming TIME is seconds since epoch)
        # We copy to avoid modifying the original dataframes
        solexs_clean = df_solexs.copy()
        
        # Astropy Time allows adding a TimeDelta (seconds) to an absolute Time
        # Convert seconds to days since Astropy TimeDelta expects days if float, or explicitly use sec
        # The safest way is to use astropy.time.TimeDelta
        from astropy.time import TimeDelta
        time_array_solexs = np.array(solexs_clean['TIME'].values, dtype=np.float64)
        solexs_td = TimeDelta(time_array_solexs, format='sec')
        solexs_clean['UTC_TIME'] = (self.solexs_epoch + solexs_td).datetime
        solexs_clean = solexs_clean.rename(columns={'FLUX': 'SOLEXS_FLUX'})
        solexs_clean = solexs_clean[['UTC_TIME', 'SOLEXS_FLUX']]
        
        # 2. Standardize HEL1OS time (assuming TIME is MJD)
        helios_clean = df_helios.copy()
        time_array_helios = np.array(helios_clean['TIME'].values, dtype=np.float64)
        helios_astropy = Time(time_array_helios, format='mjd', scale='utc')
        helios_clean['UTC_TIME'] = helios_astropy.datetime
        helios_clean = helios_clean.rename(columns={'FLUX': 'HELIOS_FLUX', 'ERROR': 'HELIOS_ERROR'})
        helios_clean = helios_clean[['UTC_TIME', 'HELIOS_FLUX', 'HELIOS_ERROR']]
        
        # Drop rows with NaT (Not a Time) just in case
        solexs_clean = solexs_clean.dropna(subset=['UTC_TIME'])
        helios_clean = helios_clean.dropna(subset=['UTC_TIME'])
        
        # Round the datetime to the nearest second to ensure clean merging
        solexs_clean['UTC_TIME'] = solexs_clean['UTC_TIME'].dt.round('1s')
        helios_clean['UTC_TIME'] = helios_clean['UTC_TIME'].dt.round('1s')
        
        # Handle duplicates that might arise from rounding by taking the mean
        solexs_clean = solexs_clean.groupby('UTC_TIME').mean().reset_index()
        helios_clean = helios_clean.groupby('UTC_TIME').mean().reset_index()
        
        # Set UTC_TIME as the index for both
        solexs_clean.set_index('UTC_TIME', inplace=True)
        helios_clean.set_index('UTC_TIME', inplace=True)
        
        # 3. Create a master 1-second timeline covering the overlapping period
        start_time = max(solexs_clean.index.min(), helios_clean.index.min())
        end_time = min(solexs_clean.index.max(), helios_clean.index.max())
        
        if start_time >= end_time or pd.isna(start_time) or pd.isna(end_time):
            raise ValueError("No overlapping time period found between SoLEXS and HEL1OS data.")
            
        master_timeline = pd.date_range(start=start_time, end=end_time, freq='1s')
        master_df = pd.DataFrame(index=master_timeline)
        master_df.index.name = 'UTC_TIME'
        
        # 4. Merge data onto the master timeline
        # This will introduce NaNs where a specific second is missing in either instrument
        # This is expected and correct - Stage 1 pipeline will handle these NaNs
        master_df = master_df.join(solexs_clean, how='left')
        master_df = master_df.join(helios_clean, how='left')
        
        return master_df.reset_index()
