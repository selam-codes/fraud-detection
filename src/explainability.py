import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract the built-in feature importances from a tree-based model
    and return the top_n features as a sorted DataFrame.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return df.sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(importance_df, title='Top Feature Importances', save_path=None):
    """
    Plot a horizontal bar chart of feature importances.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', hue='feature',
                 palette='viridis', legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig, ax


def compute_shap_values(model, X_sample):
    """
    Build a SHAP TreeExplainer for a tree-based model and compute SHAP values
    for the positive (fraud) class on X_sample.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification some SHAP/model combinations return a list of
    # per-class arrays; always return the positive-class contributions.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def find_prediction_examples(y_true, y_pred, y_proba):
    """
    Locate representative example indices (positional, 0-based) for:
    - true_positive: correctly identified fraud, with the highest confidence
    - false_positive: legitimate transaction flagged as fraud, highest confidence
    - false_negative: missed fraud, with the lowest predicted fraud probability

    Returns a dict with whichever of the three categories are present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    examples = {}
    if len(tp_idx) > 0:
        examples['true_positive'] = int(tp_idx[np.argmax(y_proba[tp_idx])])
    if len(fp_idx) > 0:
        examples['false_positive'] = int(fp_idx[np.argmax(y_proba[fp_idx])])
    if len(fn_idx) > 0:
        examples['false_negative'] = int(fn_idx[np.argmin(y_proba[fn_idx])])

    return examples
