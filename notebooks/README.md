# Notebooks

This directory contains Jupyter notebooks for exploration, feature engineering, and modeling:

1. **`eda-fraud-data.ipynb`**:
   - Performs exploratory analysis on the transaction fraud dataset.
   - Merges the IP addresses range lookup to determine the country of origin.
   - Analyzes distribution of fraud class, age, purchase value, browser, and marketing sources.
   - Identifies key geographic fraud hot spots.

2. **`eda-creditcard.ipynb`**:
   - Performs exploratory analysis on the highly-imbalanced credit card dataset.
   - Explores the PCA-reduced features, transaction amounts, and transaction timing distributions.
   - Analyzes correlations of PCA components with the target fraud class.

3. **`feature-engineering.ipynb`**:
   - Implements the feature engineering pipeline (time extractions and rolling window transaction velocity).
   - Handles train/test splitting (80/20).
   - Performs standard scaling and categorical encoding (retaining top 15 countries and mapping others to 'Other').
   - Addresses class imbalance by applying SMOTE (Synthetic Minority Over-sampling Technique) on the training split only.
   - Saves final preprocessed datasets to `data/processed/` directory.

4. **`modeling.ipynb`** *(To be implemented)*:
   - For training and evaluating ML models (e.g. Logistic Regression, Random Forest, XGBoost).

5. **`shap-explainability.ipynb`** *(To be implemented)*:
   - For model explainability using SHAP values.

## How to Run
Ensure all dependencies in `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```
To run the notebooks sequentially:
1. Load, clean, and explore datasets: Run `eda-fraud-data.ipynb` and `eda-creditcard.ipynb`.
2. Engineer features and prepare datasets: Run `feature-engineering.ipynb`.
