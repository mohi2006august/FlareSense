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
        with fits.open(file_path) as hdul:
            # Typically lightcurve data is in the first extension
            data = hdul[1].data
            
            # SoLEXS FITS contains TIME and COUNTS columns
            # COUNTS can be 1D (lightcurve) or 2D (spectral: time x channels)
            counts = data['COUNTS']
            
            if counts.ndim == 2:
                # Spectral data: sum all channels for total flux
                total_flux = np.sum(counts, axis=1)
                
                # Extract 1-5 keV band. (Assuming channels 10 to 50 map to 1-5 keV for this example)
                # In production, this channel mapping should come from the SoLEXS calibration matrix (RMF)
                flux_1_5 = np.sum(counts[:, 10:50], axis=1)
            else:
                # 1D Lightcurve data: we only have total flux. 
                # We duplicate it to FLUX_1_5KEV so the pipeline doesn't break, but issue a warning.
                print("WARNING: Loaded a 1D lightcurve file. Cannot extract 1-5 keV band. Duplicating total flux.")
                total_flux = counts
                flux_1_5 = counts
                
            df = pd.DataFrame({
                'TIME': data['TIME'],
                'FLUX': total_flux,
                'FLUX_1_5KEV': flux_1_5
            })
            
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
