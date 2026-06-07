from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def resample_data(X_train, y_train, method='smote', random_state=42):
    """
    Resample the training set to address class imbalance.
    
    Parameters:
    - X_train: Training features DataFrame
    - y_train: Training target Series
    - method: 'smote', 'undersample', or 'none'
    - random_state: Seed for reproducibility
    
    Returns:
    - X_resampled, y_resampled
    """
    print("Class distribution before resampling:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print(pd.Series(y_train).value_counts())
    
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    elif method == 'none':
        return X_train, y_train
    else:
        raise ValueError(f"Unknown resampling method: {method}")
        
    print(f"Class distribution after resampling ({method}):")
    print(pd.Series(y_resampled).value_counts(normalize=True))
    print(pd.Series(y_resampled).value_counts())
    
    return X_resampled, y_resampled
