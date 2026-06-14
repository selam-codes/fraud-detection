import unittest
import pandas as pd
from sklearn.datasets import make_classification

from src.modeling import (
    train_logistic_regression,
    tune_xgboost,
    evaluate_classifier,
    cross_validate_classifier,
)


class TestModeling(unittest.TestCase):

    def setUp(self):
        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=3, n_redundant=0,
            weights=[0.7, 0.3], random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        self.y = pd.Series(y)

    def test_train_logistic_regression(self):
        model = train_logistic_regression(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_evaluate_classifier(self):
        model = train_logistic_regression(self.X, self.y)
        result = evaluate_classifier(model, self.X, self.y, model_name='LogisticRegression')

        self.assertEqual(result['model_name'], 'LogisticRegression')
        self.assertGreaterEqual(result['auc_pr'], 0.0)
        self.assertLessEqual(result['auc_pr'], 1.0)
        self.assertGreaterEqual(result['f1_score'], 0.0)
        self.assertEqual(result['confusion_matrix'].shape, (2, 2))
        self.assertEqual(len(result['y_pred']), len(self.y))
        self.assertEqual(len(result['y_proba']), len(self.y))

    def test_cross_validate_classifier(self):
        model = train_logistic_regression(self.X, self.y)
        results = cross_validate_classifier(model, self.X, self.y, cv=3)

        for key in ['auc_pr_mean', 'auc_pr_std', 'f1_mean', 'f1_std']:
            self.assertIn(key, results)
            self.assertGreaterEqual(results[key], 0.0)

    def test_tune_xgboost(self):
        param_distributions = {'n_estimators': [10, 20], 'max_depth': [2, 3]}
        best_model, best_params, cv_results_df = tune_xgboost(
            self.X, self.y, param_distributions=param_distributions, n_iter=2, cv=2
        )

        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        self.assertFalse(cv_results_df.empty)

        preds = best_model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))


if __name__ == '__main__':
    unittest.main()
