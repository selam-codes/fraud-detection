import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from src.explainability import (
    get_feature_importance,
    compute_shap_values,
    find_prediction_examples,
)


class TestExplainability(unittest.TestCase):

    def setUp(self):
        X, y = make_classification(
            n_samples=100, n_features=4, n_informative=3, n_redundant=0,
            weights=[0.7, 0.3], random_state=42
        )
        self.X = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
        self.y = pd.Series(y)
        self.model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
        self.model.fit(self.X, self.y)

    def test_get_feature_importance(self):
        importance_df = get_feature_importance(self.model, self.X.columns.tolist(), top_n=3)

        self.assertEqual(len(importance_df), 3)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        # Sorted descending
        self.assertTrue(importance_df['importance'].is_monotonic_decreasing)

    def test_compute_shap_values(self):
        explainer, shap_values = compute_shap_values(self.model, self.X.iloc[:10])

        self.assertEqual(np.asarray(shap_values).shape[0], 10)
        self.assertEqual(np.asarray(shap_values).shape[1], self.X.shape[1])

    def test_find_prediction_examples(self):
        y_true = np.array([1, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_proba = np.array([0.9, 0.1, 0.6, 0.4, 0.8])

        examples = find_prediction_examples(y_true, y_pred, y_proba)

        self.assertEqual(examples['true_positive'], 0)
        self.assertEqual(examples['false_positive'], 2)
        self.assertEqual(examples['false_negative'], 3)


if __name__ == '__main__':
    unittest.main()
