import os
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.io import fits

class FitsDataLoader:
    """
    Loads SoLEXS and HEL1OS FITS data and associated GTI (Good Time Interval) files.
    """
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_solexs(self, file_name: str, gti_file: str = None) -> pd.DataFrame:
        """
        Load SoLEXS lightcurve/spectrum data.
        """
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"SoLEXS data file not found: {file_path}")
            
        print(f"Loading SoLEXS data from {file_path}")
        
        if str(file_path).endswith('.csv'):
            df_raw = pd.read_csv(file_path)
            # AL1_SLX_cleaned.csv has: time_s, counts, (and now counts_1_5kev)
            flux_1_5 = df_raw['counts_1_5kev'] if 'counts_1_5kev' in df_raw.columns else df_raw['counts']
            if 'counts_1_5kev' not in df_raw.columns:
                print("WARNING: Loaded a 1D CSV lightcurve. Cannot extract 1-5 keV band. Duplicating total flux.")
                
            df = pd.DataFrame({
                'TIME': df_raw['time_s'],
                'FLUX': df_raw['counts'],
                'FLUX_1_5KEV': flux_1_5
            })
        else:
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                counts = data['COUNTS']
                
                if counts.ndim == 2:
                    total_flux = np.sum(counts, axis=1)
                    flux_1_5 = np.sum(counts[:, 10:50], axis=1)
                else:
                    print("WARNING: Loaded a 1D lightcurve file. Cannot extract 1-5 keV band. Duplicating total flux.")
                    total_flux = counts
                    flux_1_5 = counts
                    
                df = pd.DataFrame({
                    'TIME': data['TIME'],
                    'FLUX': total_flux,
                    'FLUX_1_5KEV': flux_1_5
                })
        
        # Standardize time to pandas datetime if we can detect the format
        if str(file_path).endswith('.csv'):
            # UNIX epoch time in seconds
            df['TIME'] = pd.to_datetime(df['TIME'], unit='s')
            
        if gti_file:
            self._apply_gti(df, gti_file)
            
        return df
        
    def load_helios(self, file_name: str, gti_file: str = None) -> pd.DataFrame:
        """
        Load HEL1OS hard X-ray data.
        """
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"HEL1OS data file not found: {file_path}")
            
        print(f"Loading HEL1OS data from {file_path}")
        
        if str(file_path).endswith('.csv'):
            df_raw = pd.read_csv(file_path)
            # HLS_cleaned.csv has: mjd, iso_time, count_rate_per_s, stat_error
            df = pd.DataFrame({
                'TIME': pd.to_datetime(df_raw['iso_time']),
                'FLUX': df_raw['count_rate_per_s'],
                'ERROR': df_raw['stat_error']
            })
        else:
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                
                # HEL1OS FITS contains MJD, ISOT, CTR (Count Rate), STAT_ERR
                df = pd.DataFrame({
                    'TIME': data['MJD'], # We will convert this to a common format later if needed
                    'FLUX': data['CTR'],
                    'ERROR': data['STAT_ERR']
                })
            
        if gti_file:
            self._apply_gti(df, gti_file)
            
        return df
        
    def load_goes(self, file_name: str) -> pd.DataFrame:
        """
        Load GOES XRS supplementary data (CSV) for fallback patching.
        Expected columns: ['time_tag', 'flux']
        """
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"GOES data file not found: {file_path}")
            
        print(f"Loading GOES data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Standardize columns to match our internal format
        df = df.rename(columns={'time_tag': 'UTC_TIME', 'flux': 'GOES_FLUX'})
        df['UTC_TIME'] = pd.to_datetime(df['UTC_TIME'])
        
        return df
        
    def _apply_gti(self, df: pd.DataFrame, gti_file: str):
        """
        Use the GTI file to mask out bad time intervals.
        """
        gti_path = self.data_dir / gti_file
        if not gti_path.exists():
            print(f"Warning: GTI file not found: {gti_path}")
            return
            
        print(f"Applying GTI from {gti_path}")
        with fits.open(gti_path) as hdul:
            gti_data = hdul[1].data
            starts = gti_data['START']
            stops = gti_data['STOP']
            
        # Create a boolean mask: True if row['TIME'] is within ANY of the GTI intervals
        # Initialise all as False (bad)
        valid_mask = np.zeros(len(df), dtype=bool)
        
        time_array = df['TIME'].values
        for start, stop in zip(starts, stops):
            valid_mask |= (time_array >= start) & (time_array <= stop)
            
        # Set FLUX to NaN for time periods OUTSIDE the Good Time Intervals
        # Our pipeline.py will handle these NaNs (forward fill then masking token)
        df.loc[~valid_mask, 'FLUX'] = np.nan

if __name__ == "__main__":
    print("DataLoader initialized. Ready to parse Aditya-L1 datasets.")
