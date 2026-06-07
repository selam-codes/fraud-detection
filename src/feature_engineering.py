import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def extract_time_features(df):
    """
    Extracts time-based features:
    - hour_of_day: hour of purchase_time
    - day_of_week: day of purchase_time
    - time_since_signup: duration in seconds between signup_time and purchase_time
    """
    df_copy = df.copy()
    
    # Ensure datetime format
    df_copy['signup_time'] = pd.to_datetime(df_copy['signup_time'])
    df_copy['purchase_time'] = pd.to_datetime(df_copy['purchase_time'])
    
    # Time features
    df_copy['hour_of_day'] = df_copy['purchase_time'].dt.hour
    df_copy['day_of_week'] = df_copy['purchase_time'].dt.dayofweek
    df_copy['time_since_signup'] = (df_copy['purchase_time'] - df_copy['signup_time']).dt.total_seconds()
    
    return df_copy

def calculate_velocity_features(df):
    """
    Calculates transaction frequency and velocity:
    - device_sharing_count: total transactions with the same device_id
    - ip_sharing_count: total transactions with the same ip_address
    - device_tx_count_1h: rolling 1-hour transaction count for the same device_id
    - device_tx_count_24h: rolling 24-hour transaction count for the same device_id
    - ip_tx_count_1h: rolling 1-hour transaction count for the same ip_address
    - ip_tx_count_24h: rolling 24-hour transaction count for the same ip_address
    """
    df_copy = df.copy()
    
    # Ensure purchase_time is datetime
    df_copy['purchase_time'] = pd.to_datetime(df_copy['purchase_time'])
    
    # Device transaction velocity (rolling counts)
    df_time = df_copy.sort_values('purchase_time')
    
    # Perform rolling counts grouped by device_id
    dev_roll_1h = df_time.groupby('device_id').rolling('1h', on='purchase_time')['user_id'].count()
    dev_roll_24h = df_time.groupby('device_id').rolling('24h', on='purchase_time')['user_id'].count()
    
    # Perform rolling counts grouped by ip_address
    ip_roll_1h = df_time.groupby('ip_address').rolling('1h', on='purchase_time')['user_id'].count()
    ip_roll_24h = df_time.groupby('ip_address').rolling('24h', on='purchase_time')['user_id'].count()
    
    # Align and map back to device_id / purchase_time sorting
    df_dev_sorted = df_copy.sort_values(['device_id', 'purchase_time']).copy()
    df_dev_sorted['device_tx_count_1h'] = dev_roll_1h.values
    df_dev_sorted['device_tx_count_24h'] = dev_roll_24h.values
    
    # Map back to original order using the index
    df_copy['device_tx_count_1h'] = df_dev_sorted.reindex(df_copy.index)['device_tx_count_1h']
    df_copy['device_tx_count_24h'] = df_dev_sorted.reindex(df_copy.index)['device_tx_count_24h']
    
    # Align and map back to ip_address / purchase_time sorting
    df_ip_sorted = df_copy.sort_values(['ip_address', 'purchase_time']).copy()
    df_ip_sorted['ip_tx_count_1h'] = ip_roll_1h.values
    df_ip_sorted['ip_tx_count_24h'] = ip_roll_24h.values
    
    # Map back to original order using the index
    df_copy['ip_tx_count_1h'] = df_ip_sorted.reindex(df_copy.index)['ip_tx_count_1h']
    df_copy['ip_tx_count_24h'] = df_ip_sorted.reindex(df_copy.index)['ip_tx_count_24h']
    
    # Absolute sharing counts
    df_copy['device_sharing_count'] = df_copy.groupby('device_id')['user_id'].transform('count')
    df_copy['ip_sharing_count'] = df_copy.groupby('ip_address')['user_id'].transform('count')
    
    return df_copy

def preprocess_countries(train_df, test_df, top_n=15):
    """
    Identifies top_n countries in train_df. Replaces other countries in both
    train_df and test_df with 'Other'.
    """
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Get top_n countries from training set
    # (Exclude 'Unknown' if we want, but keeping it as a category is fine)
    top_countries = train_copy['country'].value_counts().head(top_n).index.tolist()
    
    # Apply to train
    train_copy['country'] = train_copy['country'].apply(lambda x: x if x in top_countries else 'Other')
    
    # Apply to test
    test_copy['country'] = test_copy['country'].apply(lambda x: x if x in top_countries else 'Other')
    
    return train_copy, test_copy

def one_hot_encode(train_df, test_df, categorical_cols):
    """
    One-hot encodes the specified categorical columns in both train and test DataFrames,
    ensuring they align perfectly in their features.
    """
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
    
    # Align columns of test set to match training set
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    
    # Convert bool columns to int
    bool_cols = train_encoded.select_dtypes(include='bool').columns
    train_encoded[bool_cols] = train_encoded[bool_cols].astype(int)
    test_encoded[bool_cols] = test_encoded[bool_cols].astype(int)
    
    return train_encoded, test_encoded

def scale_features(train_df, test_df, numerical_cols, scaler_type='standard'):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.
    Fits the scaler on train_df and transforms both train_df and test_df.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_scaled[numerical_cols] = scaler.transform(test_df[numerical_cols])
    
    return train_scaled, test_scaled, scaler
