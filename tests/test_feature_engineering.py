import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    extract_time_features,
    calculate_velocity_features,
    preprocess_countries,
    one_hot_encode,
    scale_features
)

class TestFeatureEngineering(unittest.TestCase):
    
    def test_extract_time_features(self):
        mock_data = pd.DataFrame({
            'signup_time': ['2015-01-01 00:00:00'],
            'purchase_time': ['2015-01-01 03:30:00']
        })
        
        featured = extract_time_features(mock_data)
        
        self.assertEqual(featured.loc[0, 'hour_of_day'], 3)
        self.assertEqual(featured.loc[0, 'day_of_week'], 3) # Thursday is 3 in pandas dt.dayofweek
        self.assertEqual(featured.loc[0, 'time_since_signup'], 3.5 * 3600) # 3h 30m in seconds

    def test_calculate_velocity_features(self):
        # 1 user sharing same device_id with another
        # A device_id with 2 transactions in <1h
        mock_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'device_id': ['D1', 'D1', 'D2'],
            'ip_address': [100.0, 100.0, 200.0],
            'purchase_time': [
                '2015-01-01 12:00:00',
                '2015-01-01 12:30:00', # Within 1h of user 1
                '2015-01-01 15:00:00'
            ]
        })
        
        featured = calculate_velocity_features(mock_data)
        
        # Absolute sharing
        self.assertEqual(featured.loc[0, 'device_sharing_count'], 2)
        self.assertEqual(featured.loc[1, 'device_sharing_count'], 2)
        self.assertEqual(featured.loc[2, 'device_sharing_count'], 1)
        
        # Rolling count
        self.assertEqual(featured.loc[0, 'device_tx_count_1h'], 1)
        self.assertEqual(featured.loc[1, 'device_tx_count_1h'], 2) # Row 2 is within 1h of row 1
        self.assertEqual(featured.loc[2, 'device_tx_count_1h'], 1)

    def test_preprocess_countries(self):
        train_df = pd.DataFrame({
            'country': ['USA', 'USA', 'China', 'China', 'Japan', 'France']
        })
        test_df = pd.DataFrame({
            'country': ['USA', 'Japan', 'Germany']
        })
        
        # Top 2 countries should be USA and China
        train_proc, test_proc = preprocess_countries(train_df, test_df, top_n=2)
        
        self.assertIn('USA', train_proc['country'].values)
        self.assertIn('China', train_proc['country'].values)
        # Japan and France should be 'Other'
        self.assertIn('Other', train_proc['country'].values)
        self.assertNotIn('Japan', train_proc['country'].values)
        
        # In test, USA remains USA, Germany becomes Other, Japan becomes Other (since it wasn't top 2)
        self.assertEqual(test_proc.loc[0, 'country'], 'USA')
        self.assertEqual(test_proc.loc[1, 'country'], 'Other')
        self.assertEqual(test_proc.loc[2, 'country'], 'Other')

    def test_one_hot_encode(self):
        train_df = pd.DataFrame({
            'source': ['SEO', 'Ads'],
            'sex': ['M', 'F']
        })
        test_df = pd.DataFrame({
            'source': ['SEO', 'Direct'],
            'sex': ['M', 'M']
        })
        
        train_enc, test_enc = one_hot_encode(train_df, test_df, ['source', 'sex'])
        
        # Verify columns align perfectly
        self.assertEqual(list(train_enc.columns), list(test_enc.columns))
        # Direct should not be in either, because SEO/Ads was the train set space
        # (and drop_first=True makesSEO/M drop, so Ads/F are columns)
        self.assertIn('source_SEO', train_enc.columns)
        self.assertNotIn('source_Direct', test_enc.columns)

    def test_scale_features(self):
        train_df = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
        test_df = pd.DataFrame({'val': [2.0, 4.0]})
        
        train_scaled, test_scaled, scaler = scale_features(train_df, test_df, ['val'], 'standard')
        
        # Mean of scaled train_df['val'] should be 0, std should be 1
        self.assertAlmostEqual(train_scaled['val'].mean(), 0.0)
        self.assertAlmostEqual(train_scaled['val'].std(ddof=0), 1.0)

if __name__ == '__main__':
    unittest.main()
