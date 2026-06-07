import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load a CSV dataset.
    """
    return pd.read_csv(file_path)

def clean_fraud_data(df):
    """
    Cleans the Fraud Data dataset:
    - Parses signup_time and purchase_time as datetime.
    - Handles duplicates (if any).
    - Checks for missing values.
    """
    df_copy = df.copy()
    
    # Correct datetime data types
    df_copy['signup_time'] = pd.to_datetime(df_copy['signup_time'])
    df_copy['purchase_time'] = pd.to_datetime(df_copy['purchase_time'])
    
    # Remove duplicates
    initial_len = len(df_copy)
    df_copy = df_copy.drop_duplicates()
    final_len = len(df_copy)
    if initial_len - final_len > 0:
        print(f"Removed {initial_len - final_len} duplicate rows from Fraud data.")
        
    return df_copy

def clean_creditcard_data(df):
    """
    Cleans the Credit Card dataset:
    - Removes duplicate rows.
    """
    df_copy = df.copy()
    
    # Remove duplicates
    initial_len = len(df_copy)
    df_copy = df_copy.drop_duplicates()
    final_len = len(df_copy)
    if initial_len - final_len > 0:
        print(f"Removed {initial_len - final_len} duplicate rows from Credit Card data.")
        
    return df_copy

def merge_geolocation(fraud_df, ip_df):
    """
    Merge fraud_df with ip_df based on the range-lookup constraint:
    lower_bound_ip_address <= ip_address <= upper_bound_ip_address
    
    Uses pandas.merge_asof for high-performance range mapping.
    """
    # Create copies
    fraud = fraud_df.copy()
    ip = ip_df.copy()
    
    # Convert IP addresses to float64 to ensure matching types and avoid overflows
    fraud['ip_address'] = fraud['ip_address'].astype(float)
    ip['lower_bound_ip_address'] = ip['lower_bound_ip_address'].astype(float)
    ip['upper_bound_ip_address'] = ip['upper_bound_ip_address'].astype(float)
    
    # Sort dataframes by search columns (required by merge_asof)
    fraud_sorted = fraud.sort_values('ip_address')
    ip_sorted = ip.sort_values('lower_bound_ip_address')
    
    # Merge using merge_asof: match ip_address to the closest lower_bound_ip_address <= ip_address
    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted,
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Check if the matched IP address is within the upper bound range
    in_range = merged['ip_address'] <= merged['upper_bound_ip_address']
    
    # If not in range, set country to 'Unknown'
    merged.loc[~in_range, 'country'] = np.nan
    merged['country'] = merged['country'].fillna('Unknown')
    
    # Drop lower_bound_ip_address and upper_bound_ip_address as they are no longer needed
    merged = merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    
    # Restore original order of fraud_df
    # In fraud_df, user_id is unique and can be used to reindex/resort
    # Set user_id as index and reindex to original fraud user_id order
    merged = merged.set_index('user_id').reindex(fraud['user_id']).reset_index()
    
    return merged
