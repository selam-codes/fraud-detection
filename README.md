# Fraud Detection Project - Task 1: Data Analysis and Preprocessing

This repository contains the implementation of a comprehensive fraud detection pipeline. This document outlines the Exploratory Data Analysis (EDA) report, feature engineering methodology, and resampling justification for Task 1.

---

## 1. Project Organization
The repository has been structured as follows:
```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml       # GitHub Actions workflow running unit tests
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/                    # Original datasets (Fraud_Data, IpAddress_to_Country, creditcard)
│   └── processed/              # Preprocessed, split, and SMOTE-balanced training and testing data
├── notebooks/
│   ├── __init__.py
│   ├── eda-creditcard.ipynb    # Credit Card EDA
│   ├── eda-fraud-data.ipynb    # Fraud Data EDA & Geolocation Merging
│   ├── feature-engineering.ipynb # Extraction, scaling, encoding, and SMOTE resampling
│   ├── modeling.ipynb          # Model training and selection (Placeholder)
│   ├── shap-explainability.ipynb # Model interpretation (Placeholder)
│   └── README.md
├── reports/
│   └── figures/                # Saved charts and visualizations from EDA
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py    # Cleaning, duplicate removal, IP ranges lookup
│   ├── feature_engineering.py  # Datetime features, velocity rolling, scaling, encoding
│   └── sampling.py             # SMOTE and undersampling helpers
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessor.py # Preprocessor tests
│   └── test_feature_engineering.py # Features tests
├── models/                     # Saved model artifacts
├── requirements.txt            # Package dependencies
└── README.md                   # Main documentation
```

---

## 2. Exploratory Data Analysis (EDA) Report

### A. Geolocation Integration & Lookup
Using `pandas.merge_asof` with sorted IP boundaries, we mapped transaction IP addresses in `Fraud_Data.csv` to their country in `IpAddress_to_Country.csv` in $O((N+M)\log(N+M))$ time.
- Out of 151,112 rows, **129,146** were mapped to a country.
- **21,966** rows did not fall within any designated country ranges and were filled as `'Unknown'`.
- The top country by transaction volume is the **United States** (58,049 transactions), followed by **China** (12,038) and **Japan** (7,306).

### B. Class Imbalance Metrics
Both datasets exhibit significant class imbalance:
- **Fraud Data**:
  - Legitimate: **136,966** (90.64%)
  - Fraud: **14,146** (9.36%)
- **Credit Card Data** (after removing 1,081 duplicates):
  - Legitimate: **283,253** (99.83%)
  - Fraud: **473** (0.17%)

### C. Distribution & Bivariate Relationship Insights
1. **Age and Purchase Value (Fraud Data)**: The average age is ~33 years, and the average purchase value is ~$37. The distributions of these variables are nearly identical for both fraudulent and legitimate transactions, indicating that simple linear cutoffs will not be effective for detection.
2. **Geographic Fraud Hot Spots**: While the US and China have the highest absolute volume of fraud, small-volume countries (e.g. specific nations with >50 transactions) exhibit fraud rates exceeding 20-30%.
3. **Credit Card correlations**: PCA components like `V17`, `V14`, `V12`, `V10` are strongly negatively correlated with class (lower values indicate higher fraud probability), while `V11` and `V4` show positive correlations.

---

## 3. Feature Engineering Documentation

The following features were engineered for `Fraud_Data.csv` to capture temporal and velocity behaviors:
- **Time Features**:
  - `hour_of_day`: Extraction of the transaction hour to capture night-time patterns.
  - `day_of_week`: Extraction of the transaction day to capture weekday vs. weekend patterns.
  - `time_since_signup`: Duration (in seconds) between `signup_time` and `purchase_time`. Immediate purchases (0-1 seconds) after signup are highly correlated with automatic script/bot registrations.
- **Transaction Frequency & Velocity**:
  - `device_sharing_count`: Total number of users sharing the same `device_id`. High device reuse indicates fraud networks.
  - `ip_sharing_count`: Total number of users sharing the same `ip_address`.
  - `device_tx_count_1h` & `device_tx_count_24h`: Rolling window transaction counts for the same device inside 1-hour and 24-hour windows.
  - `ip_tx_count_1h` & `ip_tx_count_24h`: Rolling window transaction counts for the same IP address.

### Categorical Encoding & Scaling
- One-hot encoding was applied to `source`, `browser`, and `sex`.
- To avoid high-dimensional sparse representations for `country`, we retained the top 15 countries by volume in the training set and mapped all remaining countries to `'Other'`.
- Standard scaling (`StandardScaler`) was applied to all continuous features, fitted *only* on the training set to prevent leakage.

---

## 4. Class Imbalance Resampling Justification

To address class imbalance without creating biased models, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the **training set only**. 

### Why SMOTE?
- **Avoids Information Loss**: Unlike random undersampling (which discards majority class instances), SMOTE preserves all majority class information, which is critical for learning the complex characteristics of legitimate behavior.
- **Reduces Overfitting**: Unlike simple random oversampling (which duplicates minority class instances, leading to severe overfitting), SMOTE generates synthetic samples along the line segments joining k-nearest neighbors of the minority class. This introduces new synthetic diversity into the training space.

### Resampling Distributions

#### Fraud Dataset (Train Split):
- **Before SMOTE**:
  - Legitimate (Class 0): **109,573** (90.64%)
  - Fraudulent (Class 1): **11,316** (9.36%)
- **After SMOTE**:
  - Legitimate (Class 0): **109,573** (50.00%)
  - Fraudulent (Class 1): **109,573** (50.00%)

#### Credit Card Dataset (Train Split):
- **Before SMOTE**:
  - Legitimate (Class 0): **226,602** (99.83%)
  - Fraudulent (Class 1): **378** (0.17%)
- **After SMOTE**:
  - Legitimate (Class 0): **226,602** (50.00%)
  - Fraudulent (Class 1): **226,602** (50.00%)

*Note: Test splits remain completely untouched, retaining their original natural distributions to ensure honest evaluation.*

---

## 5. Verification & Tests
The helper modules in `src/` are covered by unit tests in the `tests/` directory:
- Run the tests locally:
  ```bash
  python3 -m unittest discover -s tests -p "test_*.py"
  ```
- All 8 unit tests check datatypes, optimized geolocation merges, time features, velocity rolling alignments, country preprocessing, and scaling.