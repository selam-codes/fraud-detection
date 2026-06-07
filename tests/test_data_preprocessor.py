import unittest
import pandas as pd
import numpy as np
from src.data_preprocessor import clean_fraud_data, clean_creditcard_data, merge_geolocation

class TestDataPreprocessor(unittest.TestCase):
    
    def test_clean_fraud_data(self):
        # Create a mock fraud DataFrame
        mock_data = pd.DataFrame({
            'user_id': [1, 2],
            'signup_time': ['2015-01-01 00:00:00', '2015-01-02 12:00:00'],
            'purchase_time': ['2015-01-01 01:00:00', '2015-01-03 12:00:00'],
            'class': [0, 1]
        })
        
        cleaned = clean_fraud_data(mock_data)
        
        # Verify columns are datetime type
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['signup_time']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['purchase_time']))
        # Verify length is same since there are no duplicates
        self.assertEqual(len(cleaned), 2)

    def test_clean_creditcard_data_duplicates(self):
        # Create a mock creditcard DataFrame with duplicates
        mock_data = pd.DataFrame({
            'Time': [0.0, 0.0, 1.0],
            'V1': [1.5, 1.5, -2.0],
            'Class': [0, 0, 1]
        })
        
        cleaned = clean_creditcard_data(mock_data)
        # Verify duplicates are dropped
        self.assertEqual(len(cleaned), 2)

    def test_merge_geolocation(self):
        # Create a mock fraud DataFrame
        fraud_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'ip_address': [15.0, 25.0, 35.0]
        })
        
        # Create a mock IP Address to Country map
        # Range 1: 10 to 20 -> Country A
        # Range 2: 21 to 30 -> Country B
        # 35 is out of range
        ip_df = pd.DataFrame({
            'lower_bound_ip_address': [10.0, 21.0],
            'upper_bound_ip_address': [20.0, 30.0],
            'country': ['Country A', 'Country B']
        })
        
        merged = merge_geolocation(fraud_df, ip_df)
        
        # Sort by user_id to ensure order matches
        merged = merged.sort_values('user_id').reset_index(drop=True)
        
        # Verify country mapping
        self.assertEqual(merged.loc[0, 'country'], 'Country A')
        self.assertEqual(merged.loc[1, 'country'], 'Country B')
        self.assertEqual(merged.loc[2, 'country'], 'Unknown')
        
        # Check that lookup bounds were dropped
        self.assertNotIn('lower_bound_ip_address', merged.columns)
        self.assertNotIn('upper_bound_ip_address', merged.columns)

if __name__ == '__main__':
    unittest.main()
