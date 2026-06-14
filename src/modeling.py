import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train, random_state=42, max_iter=1000):
    """
    Train an interpretable Logistic Regression baseline model.
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train, param_distributions=None, n_iter=8, cv=3, random_state=42):
    """
    Perform a RandomizedSearchCV hyperparameter search for an XGBoost classifier,
    optimizing for AUC-PR (average precision).

    Returns: (best_estimator, best_params, cv_results_df)
    """
    if param_distributions is None:
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
        }

    base_model = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        eval_metric='aucpr',
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='average_precision',
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
    return search.best_estimator_, search.best_params_, cv_results_df


def evaluate_classifier(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a fitted classifier on a held-out test set using metrics
    appropriate for imbalanced data: AUC-PR, F1-Score, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'model_name': model_name,
        'auc_pr': average_precision_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=4),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


def cross_validate_classifier(model, X, y, cv=5, random_state=42):
    """
    Run Stratified K-Fold cross-validation, reporting the mean and standard
    deviation of AUC-PR and F1-Score across folds.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {'auc_pr': 'average_precision', 'f1': 'f1'}
    results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    return {
        'auc_pr_mean': results['test_auc_pr'].mean(),
        'auc_pr_std': results['test_auc_pr'].std(),
        'f1_mean': results['test_f1'].mean(),
        'f1_std': results['test_f1'].std(),
    }


def plot_confusion_matrix(cm, title='Confusion Matrix', labels=('Legitimate', 'Fraud'), ax=None, save_path=None):
    """
    Plot a confusion matrix heatmap.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return ax


def plot_pr_curves(y_test, y_proba_dict, title='Precision-Recall Curve', save_path=None):
    """
    Plot Precision-Recall curves for one or more models on the same axes.
    `y_proba_dict` maps model name -> predicted probabilities for the positive class.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f'{name} (AUC-PR = {ap:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig, ax
