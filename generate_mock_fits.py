import os
import numpy as np
from astropy.io import fits

def generate_mock_fits():
    # 1. Generate SoLEXS mock FITS (.li)
    print("Generating SoLEXS mock FITS file...")
    solexs_dir = "data/raw/SOLEXS/SDD2"
    os.makedirs(solexs_dir, exist_ok=True)
    
    # 86400 rows = 24 hours of 1-sec data
    n_rows = 86400
    # Time: seconds from epoch
    solexs_time = np.arange(n_rows, dtype=np.float64)
    # Flux: some random values
    solexs_flux = np.random.normal(loc=50.0, scale=5.0, size=n_rows)
    
    col1 = fits.Column(name='TIME', format='D', array=solexs_time)
    col2 = fits.Column(name='COUNTS', format='D', array=solexs_flux)
    cols = fits.ColDefs([col1, col2])
    
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.BinTableHDU.from_columns(cols, name='RATE')
    
    hdul_solexs = fits.HDUList([hdu1, hdu2])
    solexs_path = os.path.join(solexs_dir, "datsets2.li")
    hdul_solexs.writeto(solexs_path, overwrite=True)
    print(f"Created: {solexs_path}")
    
    # 2. Generate HEL1OS mock FITS
    print("\nGenerating HEL1OS mock FITS file...")
    helios_dir = "data/raw/HEL1OS/sdte"
    os.makedirs(helios_dir, exist_ok=True)
    
    # 43188 rows
    n_rows_helios = 43188
    # MJD starting around Jan 1 2023
    start_mjd = 59945.0
    helios_mjd = start_mjd + (np.arange(n_rows_helios, dtype=np.float64) / 86400.0)
    helios_isot = np.array(['2023-01-01T00:00:00' for _ in range(n_rows_helios)])
    helios_ctr = np.random.normal(loc=100.0, scale=10.0, size=n_rows_helios)
    helios_err = np.ones(n_rows_helios, dtype=np.float64)
    
    col_mjd = fits.Column(name='MJD', format='D', unit='MJD', array=helios_mjd)
    col_isot = fits.Column(name='ISOT', format='30A', unit='UT', array=helios_isot)
    col_ctr = fits.Column(name='CTR', format='D', unit='cts/sec', array=helios_ctr)
    col_err = fits.Column(name='STAT_ERR', format='D', unit='cts/sec', array=helios_err)
    
    cols_h = fits.ColDefs([col_mjd, col_isot, col_ctr, col_err])
    
    hdu1_h = fits.PrimaryHDU()
    hdu2_h = fits.BinTableHDU.from_columns(cols_h, name='CDTE1_LC_BAND_5.00KEV_TO_20.00KEV')
    
    hdul_helios = fits.HDUList([hdu1_h, hdu2_h])
    helios_path = os.path.join(helios_dir, "lightcurve_cdte1.fits")
    hdul_helios.writeto(helios_path, overwrite=True)
    print(f"Created: {helios_path}")

if __name__ == "__main__":
    generate_mock_fits()
