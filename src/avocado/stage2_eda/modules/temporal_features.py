import pandas as pd
import numpy as np

def calculate_temporal_features(df_fruit):
    """
    Given a DataFrame of static features for a single fruit sorted by timestamp,
    calculates longitudinal changes and speeds.
    """
    if len(df_fruit) == 0:
        return df_fruit
        
    df = df_fruit.copy()
    
    # Ensure sorted by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate elapsed hours from the first frame
    df['elapsed_hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600.0
    
    # 1. Mask-area change (percentage change relative to first frame)
    initial_area = df['mask_area'].iloc[0]
    if initial_area > 0:
        df['mask_area_change_pct'] = (df['mask_area'] - initial_area) / initial_area * 100
    else:
        df['mask_area_change_pct'] = 0.0
        
    # 2. Rolling/smoothed trends for dark coverage (window of 3 frames, min_periods=1)
    df['smoothed_dark_coverage'] = df['dark_coverage'].rolling(window=3, min_periods=1).mean()
    
    # 3. Dark-coverage change and darkening speed
    # Change from previous frame
    df['dark_coverage_diff'] = df['smoothed_dark_coverage'].diff().fillna(0)
    df['time_diff_hours'] = df['elapsed_hours'].diff().fillna(0)
    
    # Speed (change per hour)
    # Using np.where to avoid division by zero
    df['darkening_speed'] = np.where(df['time_diff_hours'] > 0, 
                                     df['dark_coverage_diff'] / df['time_diff_hours'], 
                                     0)
                                     
    # 4. Largest dark-spot growth
    df['largest_dark_spot_growth'] = df['largest_dark_spot_fraction'].diff().fillna(0)
    
    # 5. Rapid-darkening onset
    # Let's define rapid darkening as speed exceeding a certain threshold (e.g., 0.05 / hour)
    rapid_threshold = 0.01  # 1% coverage increase per hour
    df['is_rapid_darkening'] = df['darkening_speed'] > rapid_threshold
    
    # Cumulative rapid darkening flag (once it starts, it stays true)
    df['rapid_darkening_onset_flag'] = df['is_rapid_darkening'].cummax()
    
    return df
