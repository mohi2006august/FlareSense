import os
import numpy as np
import pandas as pd
from stage1.loader import FitsDataLoader
from stage1.sync import TimeSynchronizer
from astropy.io import fits

def main():
    loader = FitsDataLoader("data/raw")
    
    helios_file = "HEL1OS/sdte/lightcurve_cdte1.fits"
    print(f"Loading {helios_file}...")
    try:
        df_helios = loader.load_helios(helios_file)
        # Fix endianness from FITS
        df_helios = pd.DataFrame({col: df_helios[col].values.astype(np.float64) for col in df_helios.columns})
        print("HEL1OS DataFrame:")
        print(df_helios.head())
    except Exception as e:
        print(f"Failed to load HEL1OS: {e}")
        return

    solexs_file = "SOLEXS/SDD2/datsets2.li"
    print(f"\nLoading {solexs_file}...")
    try:
        df_solexs = loader.load_solexs(solexs_file)
        # Fix endianness from FITS
        df_solexs = pd.DataFrame({col: df_solexs[col].values.astype(np.float64) for col in df_solexs.columns})
        print("SoLEXS DataFrame:")
        print(df_solexs.head())
    except Exception as e:
        print(f"Failed to load SoLEXS: {e}")
        return

    print("\nRunning Time Synchronizer...")
    sync = TimeSynchronizer(solexs_epoch='2023-01-01T00:00:00')
    try:
        master_df = sync.sync(df_solexs, df_helios)
        print("\nSynchronized Master DataFrame (First 5 rows with data):")
        print(master_df.dropna().head(5))
        
        print("\n=== Actual FLUX values === ")
        print("SoLEXS Mean Flux:", master_df['SOLEXS_FLUX'].mean())
        print("HEL1OS Mean Flux:", master_df['HELIOS_FLUX'].mean())
        print("SoLEXS Max Flux:", master_df['SOLEXS_FLUX'].max())
        print("HEL1OS Max Flux:", master_df['HELIOS_FLUX'].max())
    except Exception as e:
        print(f"Synchronization failed: {e}")

if __name__ == "__main__":
    main()
