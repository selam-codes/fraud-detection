# Fraud Detection — Improved Detection for E-Commerce & Bank Transactions

This repository contains the full implementation of the fraud detection project for Adey Innovations Inc., covering **Task 1 (Data Analysis & Preprocessing)**, **Task 2 (Model Building & Training)**, and **Task 3 (Model Explainability with SHAP)** for two independent, highly imbalanced datasets:

- **`Fraud_Data.csv`** — e-commerce transactions (with IP-based geolocation), ~9.36% fraud.
- **`creditcard.csv`** — anonymized bank card transactions (PCA features `V1`-`V28`), ~0.17% fraud.

---

## 1. Project Organization

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml          # GitHub Actions workflow running unit tests
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/                        # Original datasets (Fraud_Data, IpAddress_to_Country, creditcard)
│   └── processed/                  # Preprocessed, split, and SMOTE-balanced train/test data
├── models/                          # Saved, trained model artifacts (joblib)
│   ├── fraud_logistic_regression.pkl
│   ├── fraud_xgboost.pkl
│   ├── credit_logistic_regression.pkl
│   └── credit_xgboost.pkl
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb        # Fraud Data EDA & Geolocation Merging (Task 1)
│   ├── eda-creditcard.ipynb        # Credit Card EDA (Task 1)
│   ├── feature-engineering.ipynb   # Feature extraction, scaling, encoding, SMOTE (Task 1)
│   ├── modeling.ipynb              # Model training, tuning, CV, comparison & selection (Task 2)
│   ├── shap-explainability.ipynb   # Feature importance & SHAP interpretation (Task 3)
│   └── README.md
├── reports/
│   └── figures/                    # Saved charts: EDA, confusion matrices, PR curves, SHAP plots
├── scripts/
│   ├── generate_notebooks.py       # Programmatically (re)generates all notebooks via nbformat
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py        # Cleaning, duplicate removal, IP range lookup
│   ├── feature_engineering.py      # Datetime features, velocity rolling, scaling, encoding
│   ├── sampling.py                 # SMOTE and undersampling helpers
│   ├── modeling.py                 # LR/XGBoost training, tuning, evaluation, CV helpers
│   └── explainability.py           # Feature importance & SHAP helpers
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessor.py
│   ├── test_feature_engineering.py
│   ├── test_modeling.py
│   └── test_explainability.py
├── generate_report.py              # Builds the consolidated PDF report (reports/)
├── requirements.txt                 # Package dependencies
└── README.md                        # Main documentation (this file)
```

---

## Task 1 — Data Analysis & Preprocessing

### 2. Exploratory Data Analysis (EDA) Report

#### A. Geolocation Integration & Lookup
Using `pandas.merge_asof` with sorted IP boundaries, we mapped transaction IP addresses in `Fraud_Data.csv` to their country in `IpAddress_to_Country.csv` in $O((N+M)\log(N+M))$ time.
- Out of 151,112 rows, **129,146** were mapped to a country.
- **21,966** rows did not fall within any designated country ranges and were filled as `'Unknown'`.
- The top country by transaction volume is the **United States** (58,049 transactions), followed by **China** (12,038) and **Japan** (7,306).

#### B. Class Imbalance Metrics
Both datasets exhibit significant class imbalance:
- **Fraud Data**:
  - Legitimate: **136,966** (90.64%)
  - Fraud: **14,146** (9.36%)
- **Credit Card Data** (after removing 1,081 duplicates):
  - Legitimate: **283,253** (99.83%)
  - Fraud: **473** (0.17%)

#### C. Distribution & Bivariate Relationship Insights
1. **Age and Purchase Value (Fraud Data)**: The average age is ~33 years, and the average purchase value is ~$37. The distributions of these variables are nearly identical for both fraudulent and legitimate transactions, indicating that simple linear cutoffs will not be effective for detection.
2. **Geographic Fraud Hot Spots**: While the US and China have the highest absolute volume of fraud, small-volume countries (e.g. specific nations with >50 transactions) exhibit fraud rates exceeding 20-30%.
3. **Credit Card correlations**: PCA components like `V17`, `V14`, `V12`, `V10` are strongly negatively correlated with class (lower values indicate higher fraud probability), while `V11` and `V4` show positive correlations.

---

### 3. Feature Engineering Documentation

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

#### Categorical Encoding & Scaling
- One-hot encoding was applied to `source`, `browser`, and `sex`.
- To avoid high-dimensional sparse representations for `country`, we retained the top 15 countries by volume in the training set and mapped all remaining countries to `'Other'`.
- Standard scaling (`StandardScaler`) was applied to all continuous features, fitted *only* on the training set to prevent leakage.

---

### 4. Class Imbalance Resampling Justification

To address class imbalance without creating biased models, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the **training set only**.

#### Why SMOTE?
- **Avoids Information Loss**: Unlike random undersampling (which discards majority class instances), SMOTE preserves all majority class information, which is critical for learning the complex characteristics of legitimate behavior.
- **Reduces Overfitting**: Unlike simple random oversampling (which duplicates minority class instances, leading to severe overfitting), SMOTE generates synthetic samples along the line segments joining k-nearest neighbors of the minority class. This introduces new synthetic diversity into the training space.

#### Resampling Distributions

**Fraud Dataset (Train Split):**
- **Before SMOTE**:
  - Legitimate (Class 0): **109,573** (90.64%)
  - Fraudulent (Class 1): **11,316** (9.36%)
- **After SMOTE**:
  - Legitimate (Class 0): **109,573** (50.00%)
  - Fraudulent (Class 1): **109,573** (50.00%)

**Credit Card Dataset (Train Split):**
- **Before SMOTE**:
  - Legitimate (Class 0): **226,602** (99.83%)
  - Fraudulent (Class 1): **378** (0.17%)
- **After SMOTE**:
  - Legitimate (Class 0): **226,602** (50.00%)
  - Fraudulent (Class 1): **226,602** (50.00%)

*Note: Test splits remain completely untouched, retaining their original natural distributions to ensure honest evaluation.*

---

## Task 2 — Model Building & Training

Implemented in `src/modeling.py` and `notebooks/modeling.ipynb`. For each dataset we trained and compared two models:

- **Logistic Regression** — an interpretable, linear baseline (`max_iter=1000`).
- **XGBoost** — a gradient-boosted tree ensemble, tuned with `RandomizedSearchCV` (optimizing **AUC-PR**, 3-fold Stratified CV).

Both models were trained on the **SMOTE-resampled training data** and evaluated on the **untouched, naturally-imbalanced test set**, since accuracy is meaningless for these class ratios. The primary metrics are **AUC-PR (average precision)** and **F1-score** on the fraud class, supplemented by confusion matrices and 5-fold Stratified K-Fold CV.

### Fraud Data (E-Commerce)

| Model | Test AUC-PR | Test F1 | Fraud Precision | Fraud Recall | CV AUC-PR (mean ± std) | CV F1 (mean ± std) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6780 | 0.5787 | 0.4976 | 0.6915 | 0.9323 ± 0.0008 | 0.8316 ± 0.0009 |
| **XGBoost (Tuned)** | **0.7066** | **0.6840** | **0.9307** | 0.5406 | 0.9893 ± 0.0001 | 0.9592 ± 0.0018 |

Best XGBoost hyperparameters: `n_estimators=100, max_depth=7, learning_rate=0.2, subsample=0.8`.

**Selected model: XGBoost (Tuned).** It improves both AUC-PR (0.678 → 0.707) and F1 (0.579 → 0.684) over the baseline. The largest practical gain is precision (49.8% → 93.1%) — XGBoost roughly halves the false-alarm rate, directly reducing the number of legitimate customers whose transactions get flagged. The trade-off is recall (69.2% → 54.1%); since XGBoost outputs a probability, the decision threshold can be tuned in production to recover recall while remaining ahead of the LR baseline at any comparable threshold.

### Credit Card Data

| Model | Test AUC-PR | Test F1 | Fraud Precision | Fraud Recall | CV AUC-PR (mean ± std) | CV F1 (mean ± std) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6750 | 0.1000 | 0.0530 | 0.8737 | 0.9921 ± 0.0002 | 0.9460 ± 0.0011 |
| **XGBoost (Tuned)** | **0.8162** | **0.7817** | 0.7549 | **0.8105** | 0.99998 ± 0.00002 | 0.99976 ± 0.00005 |

Best XGBoost hyperparameters: `n_estimators=100, max_depth=7, learning_rate=0.2, subsample=0.8`.

**Selected model: XGBoost (Tuned).** This dataset illustrates *why* a naturally-imbalanced test set is essential: both models score above 0.94 F1 under 5-fold CV on the SMOTE-resampled training data, but on the real 0.17%-fraud test set, Logistic Regression's F1 **collapses to 0.10** — at 87.4% recall, only 5.3% of its alerts are real fraud (~18 false alarms per true fraud caught). XGBoost catches almost as much fraud (81.1% recall) while being **~14x more precise** (75.5%).

Both XGBoost models (`fraud_xgboost.pkl`, `credit_xgboost.pkl`) and the Logistic Regression baselines are saved to `models/`. Full results, PR curves, and confusion matrices are in `notebooks/modeling.ipynb` and `reports/figures/`.

---

## Task 3 — Model Explainability with SHAP

Implemented in `src/explainability.py` and `notebooks/shap-explainability.ipynb`, using `shap.TreeExplainer` on each selected XGBoost model.

### Fraud Data — Top 5 SHAP Drivers

| Rank | Feature | Mean \|SHAP\| | Effect direction |
|---|---|---|---|
| 1 | `device_sharing_count` | 1.380 | Higher → **more** fraud risk (corr = +0.85) |
| 2 | `hour_of_day` | 0.456 | Non-linear / context-dependent (corr ≈ 0.00) |
| 3 | `time_since_signup` | 0.359 | Lower (recent signup) → **more** fraud risk (corr = -0.35) |
| 4 | `day_of_week` | 0.354 | Later in the week → slightly **more** fraud risk (corr = +0.32) |
| 5 | `country_United States` | 0.316 | Transacting from the US → **less** fraud risk (corr = -0.98) |

`device_sharing_count` is #1 in both SHAP and built-in (gain) importance — strong agreement that device-sharing across accounts is the dominant fraud signal. Notably, `device_tx_count_1h` is the **#2 built-in feature** (gain importance 0.274) but doesn't appear in SHAP's top 10 — it's a *rare-but-decisive* signal that fires hard for a few high-velocity bursts but is near-zero for most transactions. Conversely, `hour_of_day` ranks #2 by SHAP yet has almost zero linear correlation with its own SHAP value (corr = -0.045) — its effect is driven by *interactions* with other features, not a simple "fraud happens at night" rule.

### Credit Card Data — Top 5 SHAP Drivers

| Rank | Feature | Mean \|SHAP\| | Effect direction |
|---|---|---|---|
| 1 | `V14` | 2.051 | Lower V14 → **more** fraud risk (corr = -0.90) |
| 2 | `V4`  | 1.554 | Higher V4 → **more** fraud risk (corr = +0.86) |
| 3 | `V12` | 0.801 | Lower V12 → **more** fraud risk (corr = -0.85) |
| 4 | `V11` | 0.752 | Higher V11 → **more** fraud risk (corr = +0.79) |
| 5 | `V8`  | 0.732 | Lower V8 → **more** fraud risk (corr = -0.54) |

SHAP and gain-based importance **agree closely** here — `V14` and `V4` rank #1/#2 in both (`V14` alone = 63.3% of gain importance), consistent with the Task 1 EDA finding that `V17`, `V14`, `V12`, `V10` correlate most strongly with `Class`. Because `V1`-`V28` are anonymized PCA components, the model's top driver is a latent combination of original features invisible to a human analyst — which is precisely why SHAP is essential even for a high-performing model.

### Individual Prediction Analysis (Force Plots)

For both datasets we inspected the highest-confidence True Positive, the highest-confidence False Positive, and the lowest-confidence False Negative:

- **Fraud TP**: `ip_tx_count_1h` (+3.62), `time_since_signup` (+3.06), `device_sharing_count` (+2.31) — a textbook automated "bot-network" signature.
- **Fraud FP**: nearly identical `time_since_signup` (+3.34) and `device_sharing_count` (+2.52) contributions to the TP case — a legitimate customer on a shared device making a fast first purchase, indistinguishable from the bot pattern using these features alone.
- **Fraud FN**: `device_sharing_count` (-3.31) and `country_Germany` (-1.87) push hard toward "legitimate", masking a fraud case that doesn't rely on device-sharing or velocity at all.
- **Credit TP**: `V14` (+6.86), `V10` (+1.84), `V17` (+1.51) — an extreme multi-dimensional PCA outlier.
- **Credit FP**: `V14` (+6.98) and `V10` (+1.85) are *even larger* than the TP case — a legitimate transaction that happens to share the same outlier profile.
- **Credit FN**: every top driver (`V8` -2.59, `V14` -2.11, `V11` -1.80) pushes toward "legitimate" — a fraud case that simply doesn't look like a PCA outlier, so it passes through undetected.

Full plots are saved in `reports/figures/` (`*_shap_summary.png`, `*_shap_force_*.png`, `*_feature_importance.png`).

### Business Recommendations

1. **Add step-up verification for "fast first purchase" transactions.** `time_since_signup` is a top-3 SHAP driver, and a purchase within minutes of signup strongly raises the fraud score in both the TP and FP examples. Require additional verification (email/SMS confirmation, temporary spending cap) for purchases made shortly after account creation, instead of outright blocking — this preserves the signal while reducing friction for genuinely new customers.
2. **Treat device/IP-sharing as a risk-scoring input, not an automatic block.** `device_sharing_count` is the #1 driver overall, but the FP example shows legitimate shared-device households trigger it too. Use it to raise a risk score and route to manual review/step-up auth rather than auto-decline.
3. **Invest in complementary signals for "low-velocity" fraud.** The FN example shows fraud that doesn't exhibit device-sharing or velocity patterns is currently missed entirely. Behavioral biometrics, payment-instrument reputation, or compromised-credential cross-referencing would target this blind spot that velocity features cannot see.
4. **Monitor `V14`, `V4`, `V10`, `V12`, `V11` as a composite "outlier score" for the credit card stream.** These five components dominate both gain-based and SHAP importance; surfacing a simple composite outlier score to the fraud-review team gives a fast triage signal alongside the full model score.
5. **Set the operating threshold from the Precision-Recall curve, not a default 0.5 cutoff.** Logistic Regression's 5.3% precision on the credit card data (≈18 false alarms per true fraud) would overwhelm any review team. XGBoost's PR curve keeps precision above ~75% out to ~80% recall — choose the threshold based on the team's realistic review capacity, and re-validate periodically.
6. **Re-run SHAP analysis periodically as a model-monitoring tool.** Shifts in the ranking or magnitude of top SHAP drivers (e.g., `device_sharing_count`'s share of total impact dropping) can serve as an early warning that fraud patterns have changed and the model needs retraining.

---

## Verification & Tests

The helper modules in `src/` are covered by **15 unit tests** across `tests/`:
- `test_data_preprocessor.py` — data cleaning, duplicate removal, geolocation merge.
- `test_feature_engineering.py` — time features, velocity rolling, country preprocessing, scaling, encoding.
- `test_modeling.py` — Logistic Regression training, XGBoost tuning, evaluation, cross-validation.
- `test_explainability.py` — feature importance extraction, SHAP value computation, prediction-example selection.

Run all tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## Reproducing the Notebooks

All notebooks are generated programmatically via `scripts/generate_notebooks.py` (using `nbformat`) and then executed with `jupyter nbconvert`:
```bash
python scripts/generate_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/shap-explainability.ipynb
```

The final consolidated PDF report is built with:
```bash
python generate_report.py
```
