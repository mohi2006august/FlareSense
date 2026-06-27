import pandas as pd
import numpy as np
from typing import Tuple

class FlareLabelGenerator:
    """
    Generates ground-truth multi-class labels for 6-hour windows by cross-referencing
    the window timestamps against a GOES supplementary flare catalog.
    
    Classes:
    0 = Background / Quiet Sun
    1 = C-Class Flare
    2 = M-Class Flare
    3 = X-Class Flare
    """
    
    CLASS_MAP = {'C': 1, 'M': 2, 'X': 3}
    
    def __init__(self, catalog_path: str):
        """
        Loads the supplementary GOES flare catalog (CSV).
        Expected columns: ['start_time', 'peak_time', 'end_time', 'class']
        """
        self.catalog = pd.read_csv(catalog_path)
        self.catalog['start_time'] = pd.to_datetime(self.catalog['start_time'])
        self.catalog['end_time'] = pd.to_datetime(self.catalog['end_time'])
        
        # Extract the base class letter (C, M, X) and map to integer
        # e.g., 'M5.2' -> 'M' -> 2
        self.catalog['class_int'] = self.catalog['class'].str[0].map(self.CLASS_MAP).fillna(0).astype(int)
        
    def get_label_for_window(self, window_start: pd.Timestamp, window_end: pd.Timestamp) -> Tuple[int, float]:
        """
        Takes the start and end of a 6-hour window and returns the label.
        If multiple flares occur, returns the most severe one.
        
        Returns:
            label (int): 0, 1, 2, or 3
            onset_time (float): Hours from the start of the window to the flare peak 
                                (or 0.0 if no flare).
        """
        # Find any flares that overlap with our 6-hour window
        # Overlap condition: flare_start < window_end AND flare_end > window_start
        overlapping_flares = self.catalog[
            (self.catalog['start_time'] < window_end) & 
            (self.catalog['end_time'] > window_start)
        ]
        
        if overlapping_flares.empty:
            return 0, 0.0  # Background class
            
        # Get the most severe flare in this window
        max_flare = overlapping_flares.loc[overlapping_flares['class_int'].idxmax()]
        
        label = max_flare['class_int']
        
        # Calculate onset time in hours relative to the start of the window
        # We use peak_time if available, otherwise start_time
        flare_time = max_flare.get('peak_time', max_flare['start_time'])
        flare_time = pd.to_datetime(flare_time)
        
        onset_hours = (flare_time - window_start).total_seconds() / 3600.0
        
        # Clip onset to window boundaries (0 to 6 hours) just in case
        onset_hours = np.clip(onset_hours, 0.0, 6.0)
        
        return label, onset_hours
