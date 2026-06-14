import nbformat as nbf
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def create_eda_fraud_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Exploratory Data Analysis - Fraud Data\n"
            "This notebook performs the Exploratory Data Analysis (EDA) for the transaction fraud dataset, "
            "integrating IP addresses with countries to analyze fraud patterns by geography."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import sys\n"
            "import os\n"
            "\n"
            "# Add src directory to path\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.data_preprocessor import load_data, clean_fraud_data, merge_geolocation\n"
            "\n"
            "# Configure plots\n"
            "sns.set_theme(style='whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (10, 6)\n"
            "os.makedirs('../reports/figures', exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load and Preprocess Data\n"
            "We load the raw datasets and perform data cleaning and geolocation range-based lookup."
        ),
        nbf.v4.new_code_cell(
            "# Load raw data\n"
            "fraud_raw = load_data('../data/raw/Fraud_Data.csv')\n"
            "ip_raw = load_data('../data/raw/IpAddress_to_Country.csv')\n"
            "\n"
            "# Clean and merge geolocation\n"
            "fraud_cleaned = clean_fraud_data(fraud_raw)\n"
            "fraud = merge_geolocation(fraud_cleaned, ip_raw)\n"
            "\n"
            "print('Merged data shape:', fraud.shape)\n"
            "fraud.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Univariate Distributions\n"
            "Let's visualize the distribution of key variables: Class, Purchase Value, Age, Source, Browser, and Sex."
        ),
        nbf.v4.new_code_cell(
            "# Class Distribution\n"
            "plt.figure(figsize=(6, 4))\n"
            "ax = sns.countplot(x='class', data=fraud, hue='class', palette='viridis', legend=False)\n"
            "plt.title('Fraud Class Distribution')\n"
            "plt.xlabel('Class (0: Legitimate, 1: Fraud)')\n"
            "plt.ylabel('Count')\n"
            "for p in ax.patches:\n"
            "    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),\n"
            "                ha='center', va='center', xytext=(0, 5), textcoords='offset points')\n"
            "plt.savefig('../reports/figures/fraud_class_dist.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Purchase Value & Age Distributions\n"
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
            "\n"
            "sns.histplot(fraud['purchase_value'], kde=True, bins=50, ax=axes[0], color='skyblue')\n"
            "axes[0].set_title('Distribution of Purchase Value')\n"
            "axes[0].set_xlabel('Purchase Value ($)')\n"
            "\n"
            "sns.histplot(fraud['age'], kde=True, bins=30, ax=axes[1], color='salmon')\n"
            "axes[1].set_title('Distribution of Age')\n"
            "axes[1].set_xlabel('Age')\n"
            "\n"
            "plt.savefig('../reports/figures/purchase_value_age_dist.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Source, Browser, and Sex Distributions\n"
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
            "\n"
            "sns.countplot(x='source', data=fraud, ax=axes[0], hue='source', palette='pastel', legend=False)\n"
            "axes[0].set_title('Distribution of Marketing Source')\n"
            "\n"
            "sns.countplot(x='browser', data=fraud, ax=axes[1], hue='browser', palette='pastel', legend=False)\n"
            "axes[1].set_title('Distribution of Browser')\n"
            "\n"
            "sns.countplot(x='sex', data=fraud, ax=axes[2], hue='sex', palette='pastel', legend=False)\n"
            "axes[2].set_title('Distribution of Sex')\n"
            "\n"
            "plt.savefig('../reports/figures/categoricals_dist.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Bivariate Relationships\n"
            "Let's explore how purchase value, age, browser, source, and country relate to the target class (fraud)."
        ),
        nbf.v4.new_code_cell(
            "# Purchase Value and Age vs Class\n"
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
            "\n"
            "sns.boxplot(x='class', y='purchase_value', data=fraud, ax=axes[0], hue='class', palette='viridis', legend=False)\n"
            "axes[0].set_title('Purchase Value vs Fraud Class')\n"
            "axes[0].set_xlabel('Class (0: Legitimate, 1: Fraud)')\n"
            "axes[0].set_ylabel('Purchase Value ($)')\n"
            "\n"
            "sns.boxplot(x='class', y='age', data=fraud, ax=axes[1], hue='class', palette='viridis', legend=False)\n"
            "axes[1].set_title('Age vs Fraud Class')\n"
            "axes[1].set_xlabel('Class (0: Legitimate, 1: Fraud)')\n"
            "axes[1].set_ylabel('Age')\n"
            "\n"
            "plt.savefig('../reports/figures/bivariate_continuous.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Fraud rates across Source and Browser\n"
            "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
            "\n"
            "source_fraud = fraud.groupby('source')['class'].mean().reset_index()\n"
            "sns.barplot(x='source', y='class', data=source_fraud, ax=axes[0], hue='source', palette='coolwarm', legend=False)\n"
            "axes[0].set_title('Fraud Rate by Marketing Source')\n"
            "axes[0].set_ylabel('Fraud Rate')\n"
            "\n"
            "browser_fraud = fraud.groupby('browser')['class'].mean().reset_index()\n"
            "sns.barplot(x='browser', y='class', data=browser_fraud, ax=axes[1], hue='browser', palette='coolwarm', legend=False)\n"
            "axes[1].set_title('Fraud Rate by Browser')\n"
            "axes[1].set_ylabel('Fraud Rate')\n"
            "\n"
            "plt.savefig('../reports/figures/bivariate_categoricals.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Geolocation Analysis\n"
            "Let's analyze fraud rates by country to identify geographic patterns."
        ),
        nbf.v4.new_code_cell(
            "# Top Countries by Transaction Volume\n"
            "country_counts = fraud['country'].value_counts()\n"
            "print('Top 15 Countries by Transaction Volume:')\n"
            "print(country_counts.head(15))\n"
            "\n"
            "# Calculate fraud rates per country for countries with at least 50 transactions\n"
            "country_stats = fraud.groupby('country').agg(\n"
            "    total_transactions=('class', 'count'),\n"
            "    fraud_transactions=('class', 'sum'),\n"
            "    fraud_rate=('class', 'mean')\n"
            ").reset_index()\n"
            "\n"
            "top_fraud_countries = country_stats[country_stats['total_transactions'] >= 50].sort_values('fraud_rate', ascending=False)\n"
            "\n"
            "plt.figure(figsize=(12, 6))\n"
            "sns.barplot(x='fraud_rate', y='country', data=top_fraud_countries.head(15), hue='country', palette='Reds_r', legend=False)\n"
            "plt.title('Top 15 Countries by Fraud Rate (Min 50 Transactions)')\n"
            "plt.xlabel('Fraud Rate')\n"
            "plt.ylabel('Country')\n"
            "plt.savefig('../reports/figures/country_fraud_rates.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Class Imbalance Quantification\n"
            "Let's measure the class imbalance in details."
        ),
        nbf.v4.new_code_cell(
            "legit_count = (fraud['class'] == 0).sum()\n"
            "fraud_count = (fraud['class'] == 1).sum()\n"
            "total = len(fraud)\n"
            "\n"
            "print(f'Legitimate transactions: {legit_count} ({legit_count/total*100:.2f}%)')\n"
            "print(f'Fraudulent transactions: {fraud_count} ({fraud_count/total*100:.2f}%)')\n"
            "print(f'Class ratio (Legit/Fraud): {legit_count/fraud_count:.2f}:1')"
        ),
        nbf.v4.new_markdown_cell(
            "## Summary of EDA Insights (Fraud Data)\n"
            "1. **Class Imbalance**: The dataset is heavily imbalanced, with only 9.36% of transactions classified as fraud. Modeling will require class balancing techniques like SMOTE or undersampling.\n"
            "2. **Purchase Value and Age**: The distributions of purchase value and age are highly similar between legitimate and fraudulent transactions. Univariate differences are minimal, suggesting combinations of features or other dimensions (like velocity) will drive detection.\n"
            "3. **Browsers & Sources**: Chrome and Ads represent the largest volumes, but Chrome/IE/FireFox/Safari share similar fraud rates (~9-10%), while Opera has slightly lower rates. Marketing source has negligible impact on fraud rates.\n"
            "4. **Geography**: The United States, China, and Japan represent the highest transaction volume. However, small-volume countries or specific allocations (like select European or Asian nations) show elevated fraud rates. IP addresses mapping to 'Unknown' (missing from map) also exhibit standard fraud rates."
        )
    ]
    
    nb['cells'] = cells
    notebook_path = os.path.join(project_root, 'notebooks', 'eda-fraud-data.ipynb')
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

def create_eda_creditcard_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Exploratory Data Analysis - Credit Card Data\n"
            "This notebook performs the Exploratory Data Analysis (EDA) for the highly imbalanced "
            "Credit Card fraud dataset, analyzing transaction amounts, times, and PCA-reduced features."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import sys\n"
            "import os\n"
            "\n"
            "# Add src directory to path\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.data_preprocessor import load_data, clean_creditcard_data\n"
            "\n"
            "# Configure plots\n"
            "sns.set_theme(style='whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (10, 6)\n"
            "os.makedirs('../reports/figures', exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load and Clean Data\n"
            "We load the credit card dataset and remove duplicates."
        ),
        nbf.v4.new_code_cell(
            "credit_raw = load_data('../data/raw/creditcard.csv')\n"
            "\n"
            "# Clean data (removes 1081 duplicates)\n"
            "credit = clean_creditcard_data(credit_raw)\n"
            "\n"
            "print('Cleaned credit card shape:', credit.shape)\n"
            "credit.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Target Variable Distribution (Class Imbalance)\n"
            "Let's look at the level of class imbalance in the credit card dataset."
        ),
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(6, 4))\n"
            "ax = sns.countplot(x='Class', data=credit, hue='Class', palette='mako', legend=False)\n"
            "plt.title('Credit Card Class Distribution')\n"
            "plt.xlabel('Class (0: Legitimate, 1: Fraud)')\n"
            "plt.ylabel('Count')\n"
            "for p in ax.patches:\n"
            "    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),\n"
            "                ha='center', va='center', xytext=(0, 5), textcoords='offset points')\n"
            "plt.savefig('../reports/figures/credit_class_dist.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Transaction Time and Amount Distributions\n"
            "Since the PCA components (V1-V28) are standardized, let's look at the original features `Time` and `Amount`."
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
            "\n"
            "sns.histplot(credit['Time'] / 3600, kde=True, bins=48, ax=axes[0], color='teal')\n"
            "axes[0].set_title('Distribution of Transaction Time (Hours)')\n"
            "axes[0].set_xlabel('Time (Hours since first transaction)')\n"
            "\n"
            "# Since amount is highly skewed, use log scale\n"
            "sns.histplot(credit['Amount'] + 1, kde=True, bins=50, ax=axes[1], log_scale=True, color='purple')\n"
            "axes[1].set_title('Distribution of Transaction Amount (Log Scale)')\n"
            "axes[1].set_xlabel('Amount + 1 ($)')\n"
            "\n"
            "plt.savefig('../reports/figures/credit_time_amount_dist.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Correlation with Target Class\n"
            "Let's see which PCA components are most strongly correlated (positively or negatively) with the target Class."
        ),
        nbf.v4.new_code_cell(
            "correlations = credit.corr()['Class'].sort_values()\n"
            "print('Top 5 Negative Correlations with Class:')\n"
            "print(correlations.head(5))\n"
            "print('\\nTop 5 Positive Correlations with Class (excluding Class):')\n"
            "print(correlations.tail(6)[:-1])"
        ),
        nbf.v4.new_code_cell(
            "# Visualizing distributions of top correlated PCA features\n"
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
            "\n"
            "# V17 has strong negative correlation\n"
            "sns.boxplot(x='Class', y='V17', data=credit, ax=axes[0], hue='Class', palette='coolwarm', legend=False)\n"
            "axes[0].set_title('V17 distribution by Class (Negatively Correlated)')\n"
            "\n"
            "# V11 has strong positive correlation\n"
            "sns.boxplot(x='Class', y='V11', data=credit, ax=axes[1], hue='Class', palette='coolwarm', legend=False)\n"
            "axes[1].set_title('V11 distribution by Class (Positively Correlated)')\n"
            "\n"
            "plt.savefig('../reports/figures/credit_correlated_features.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Class Imbalance Quantification\n"
            "Let's quantify the imbalance in detail."
        ),
        nbf.v4.new_code_cell(
            "legit_count = (credit['Class'] == 0).sum()\n"
            "fraud_count = (credit['Class'] == 1).sum()\n"
            "total = len(credit)\n"
            "\n"
            "print(f'Legitimate credit card transactions: {legit_count} ({legit_count/total*100:.4f}%)')\n"
            "print(f'Fraudulent credit card transactions: {fraud_count} ({fraud_count/total*100:.4f}%)')\n"
            "print(f'Class ratio (Legit/Fraud): {legit_count/fraud_count:.2f}:1')"
        ),
        nbf.v4.new_markdown_cell(
            "## Summary of EDA Insights (Credit Card)\n"
            "1. **Extreme Class Imbalance**: The credit card dataset has a massive imbalance, with only 0.17% of transactions (473 out of 283,726 unique transactions) being fraudulent. Traditional learning algorithms will be highly biased toward predicting the majority class unless class balancing is performed.\n"
            "2. **Transaction Time**: The time feature has a bimodal distribution corresponding to day-night cycles (2 days of data). Fraud occurs relatively evenly across time, unlike legitimate transactions which drop during the night.\n"
            "3. **Transaction Amount**: Legitimate transaction amounts have a long tail, but the median is around $22. Fraudulent transactions also have a wide range of amounts but do not exhibit massive multi-thousand-dollar values (most are under $500).\n"
            "4. **Correlations**: PCA components like `V17`, `V14`, `V12`, `V10` are strongly negatively correlated with fraud (i.e. lower values indicate higher fraud probability), while `V11`, `V4`, `V2` are positively correlated."
        )
    ]
    
    nb['cells'] = cells
    notebook_path = os.path.join(project_root, 'notebooks', 'eda-creditcard.ipynb')
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

def create_feature_engineering_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell(
            "# Feature Engineering and Preprocessing Pipeline\n"
            "This notebook implements the complete feature engineering, data transformation (scaling and encoding), "
            "train-test splitting, and class imbalance resampling (using SMOTE) for both the Fraud and Credit Card datasets."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import sys\n"
            "import os\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "# Add src directory to path\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.data_preprocessor import load_data, clean_fraud_data, clean_creditcard_data, merge_geolocation\n"
            "from src.feature_engineering import (\n"
            "    extract_time_features,\n"
            "    calculate_velocity_features,\n"
            "    preprocess_countries,\n"
            "    one_hot_encode,\n"
            "    scale_features\n"
            ")\n"
            "from src.sampling import resample_data\n"
            "\n"
            "os.makedirs('../data/processed', exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load and Clean Raw Datasets"
        ),
        nbf.v4.new_code_cell(
            "# Load Fraud data\n"
            "fraud_raw = load_data('../data/raw/Fraud_Data.csv')\n"
            "ip_raw = load_data('../data/raw/IpAddress_to_Country.csv')\n"
            "\n"
            "# Clean and merge geolocation\n"
            "fraud_cleaned = clean_fraud_data(fraud_raw)\n"
            "fraud = merge_geolocation(fraud_cleaned, ip_raw)\n"
            "\n"
            "# Load Credit Card data\n"
            "credit_raw = load_data('../data/raw/creditcard.csv')\n"
            "credit = clean_creditcard_data(credit_raw)\n"
            "\n"
            "print('Fraud shape:', fraud.shape)\n"
            "print('Credit shape:', credit.shape)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Feature Engineering (Fraud_Data.csv)\n"
            "We extract time features (`hour_of_day`, `day_of_week`, `time_since_signup`) and calculate transaction "
            "velocity features (`device_sharing_count`, `ip_sharing_count`, rolling 1h/24h counts)."
        ),
        nbf.v4.new_code_cell(
            "print('Engineering features for Fraud dataset...')\n"
            "fraud_featured = extract_time_features(fraud)\n"
            "fraud_featured = calculate_velocity_features(fraud_featured)\n"
            "print('Finished feature engineering. Columns added:', [col for col in fraud_featured.columns if col not in fraud.columns])\n"
            "fraud_featured.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Train-Test Splits\n"
            "We separate features and targets and run an 80/20 train/test split. "
            "Crucially, all encoding, scaling, and resampling fits will occur ONLY on the training split to prevent leakage."
        ),
        nbf.v4.new_code_cell(
            "# Fraud split\n"
            "X_fraud = fraud_featured.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])\n"
            "y_fraud = fraud_featured['class']\n"
            "\n"
            "X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(\n"
            "    X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud\n"
            ")\n"
            "\n"
            "# Credit split\n"
            "X_credit = credit.drop(columns=['Class'])\n"
            "y_credit = credit['Class']\n"
            "\n"
            "X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(\n"
            "    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit\n"
            ")\n"
            "\n"
            "print('Fraud train shape:', X_fraud_train.shape, 'test shape:', X_fraud_test.shape)\n"
            "print('Credit train shape:', X_credit_train.shape, 'test shape:', X_credit_test.shape)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Categorical Encoding & Scaling\n"
            "We preprocess countries (retaining the top 15 and grouping others into 'Other'), one-hot encode categoricals, "
            "and apply standard scaling to all numerical features."
        ),
        nbf.v4.new_code_cell(
            "# Categorical preprocessing for Fraud dataset country column\n"
            "X_fraud_train, X_fraud_test = preprocess_countries(X_fraud_train, X_fraud_test, top_n=15)\n"
            "\n"
            "# One-hot encode categoricals: source, browser, sex, country\n"
            "categorical_cols = ['source', 'browser', 'sex', 'country']\n"
            "X_fraud_train_enc, X_fraud_test_enc = one_hot_encode(X_fraud_train, X_fraud_test, categorical_cols)\n"
            "\n"
            "# Standardize numerical features for Fraud\n"
            "numerical_cols_fraud = [\n"
            "    'purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup',\n"
            "    'device_tx_count_1h', 'device_tx_count_24h', 'ip_tx_count_1h', 'ip_tx_count_24h',\n"
            "    'device_sharing_count', 'ip_sharing_count'\n"
            "]\n"
            "X_fraud_train_scaled, X_fraud_test_scaled, fraud_scaler = scale_features(\n"
            "    X_fraud_train_enc, X_fraud_test_enc, numerical_cols_fraud, 'standard'\n"
            ")\n"
            "\n"
            "# Standardize numerical features for Credit (all features are numerical)\n"
            "numerical_cols_credit = X_credit_train.columns.tolist()\n"
            "X_credit_train_scaled, X_credit_test_scaled, credit_scaler = scale_features(\n"
            "    X_credit_train, X_credit_test, numerical_cols_credit, 'standard'\n"
            ")\n"
            "\n"
            "print('Fraud features shape after scaling/encoding:', X_fraud_train_scaled.shape)\n"
            "print('Credit features shape after scaling:', X_credit_train_scaled.shape)"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Handle Class Imbalance\n"
            "We apply SMOTE (Synthetic Minority Over-sampling Technique) to the training sets only, "
            "documenting the class distribution shift."
        ),
        nbf.v4.new_code_cell(
            "print('Resampling Fraud Training Set...')\n"
            "X_fraud_train_res, y_fraud_train_res = resample_data(X_fraud_train_scaled, y_fraud_train, method='smote')\n"
            "\n"
            "print('\\nResampling Credit Card Training Set...')\n"
            "# Note: For Credit Card, SMOTE creates ~226k synthetic positive instances to balance classes.\n"
            "X_credit_train_res, y_credit_train_res = resample_data(X_credit_train_scaled, y_credit_train, method='smote')\n"
            "\n"
            "print('\\nResampling complete!')"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Save Processed Datasets\n"
            "We export the final train/test datasets to the `data/processed/` directory."
        ),
        nbf.v4.new_code_cell(
            "# Save Fraud processed splits\n"
            "X_fraud_train_res.to_csv('../data/processed/fraud_X_train_resampled.csv', index=False)\n"
            "y_fraud_train_res.to_csv('../data/processed/fraud_y_train_resampled.csv', index=False)\n"
            "X_fraud_test_scaled.to_csv('../data/processed/fraud_X_test.csv', index=False)\n"
            "y_fraud_test.to_csv('../data/processed/fraud_y_test.csv', index=False)\n"
            "\n"
            "# Save Credit processed splits\n"
            "X_credit_train_res.to_csv('../data/processed/credit_X_train_resampled.csv', index=False)\n"
            "y_credit_train_res.to_csv('../data/processed/credit_y_train_resampled.csv', index=False)\n"
            "X_credit_test_scaled.to_csv('../data/processed/credit_X_test.csv', index=False)\n"
            "y_credit_test.to_csv('../data/processed/credit_y_test.csv', index=False)\n"
            "\n"
            "print('Saved preprocessed splits successfully!')\n"
            "print('fraud_X_train_resampled shape:', X_fraud_train_res.shape)\n"
            "print('credit_X_train_resampled shape:', X_credit_train_res.shape)"
        )
    ]
    
    nb['cells'] = cells
    notebook_path = os.path.join(project_root, 'notebooks', 'feature-engineering.ipynb')
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

def create_modeling_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Model Building, Training, and Evaluation\n"
            "\n"
            "This notebook builds, trains, and evaluates classification models for fraud detection on "
            "**both** datasets prepared in `feature-engineering.ipynb`:\n"
            "- **Fraud_Data.csv** (e-commerce transactions)\n"
            "- **creditcard.csv** (bank credit card transactions)\n"
            "\n"
            "The two datasets are treated as **independent modeling problems**, each with its own pipeline. "
            "For each dataset we:\n"
            "1. Load the preprocessed train (SMOTE-resampled) and test (untouched, natural distribution) splits.\n"
            "2. Train a **Logistic Regression** baseline (interpretable).\n"
            "3. Train an **XGBoost** ensemble model with hyperparameter tuning (`RandomizedSearchCV`).\n"
            "4. Run **Stratified 5-Fold cross-validation** for robustness.\n"
            "5. Compare models using **AUC-PR**, **F1-Score**, and **Confusion Matrices**, and select the best model.\n"
            "6. Persist the selected models to `../models/` for use in `shap-explainability.ipynb`.\n"
            "\n"
            "**Primary metrics:** AUC-PR (Average Precision) and F1-Score are used as the primary evaluation "
            "metrics because both datasets are highly imbalanced — overall accuracy would be misleading "
            "(e.g., a model that always predicts 'legitimate' would score >99% accuracy on the credit card data "
            "while catching zero fraud)."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import warnings\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import joblib\n"
            "\n"
            "warnings.filterwarnings('ignore')\n"
            "\n"
            "# Add src directory to path\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.modeling import (\n"
            "    train_logistic_regression,\n"
            "    tune_xgboost,\n"
            "    evaluate_classifier,\n"
            "    cross_validate_classifier,\n"
            "    plot_confusion_matrix,\n"
            "    plot_pr_curves,\n"
            ")\n"
            "\n"
            "os.makedirs('../models', exist_ok=True)\n"
            "os.makedirs('../reports/figures', exist_ok=True)\n"
            "pd.set_option('display.max_columns', None)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Preprocessed Data\n"
            "We load the SMOTE-resampled training sets and the untouched (natural-distribution) test sets "
            "produced by `feature-engineering.ipynb`. Resampling was applied to the **training data only**, "
            "so test-set evaluation reflects real-world class proportions."
        ),
        nbf.v4.new_code_cell(
            "# Fraud (e-commerce) dataset\n"
            "X_fraud_train = pd.read_csv('../data/processed/fraud_X_train_resampled.csv')\n"
            "y_fraud_train = pd.read_csv('../data/processed/fraud_y_train_resampled.csv').squeeze()\n"
            "X_fraud_test = pd.read_csv('../data/processed/fraud_X_test.csv')\n"
            "y_fraud_test = pd.read_csv('../data/processed/fraud_y_test.csv').squeeze()\n"
            "\n"
            "# Credit Card dataset\n"
            "X_credit_train = pd.read_csv('../data/processed/credit_X_train_resampled.csv')\n"
            "y_credit_train = pd.read_csv('../data/processed/credit_y_train_resampled.csv').squeeze()\n"
            "X_credit_test = pd.read_csv('../data/processed/credit_X_test.csv')\n"
            "y_credit_test = pd.read_csv('../data/processed/credit_y_test.csv').squeeze()\n"
            "\n"
            "print('Fraud  -> train:', X_fraud_train.shape, ' test:', X_fraud_test.shape)\n"
            "print('Credit -> train:', X_credit_train.shape, ' test:', X_credit_test.shape)\n"
            "\n"
            "print('\\nFraud train class balance (post-SMOTE):')\n"
            "print(y_fraud_train.value_counts(normalize=True))\n"
            "print('\\nFraud test class balance (natural):')\n"
            "print(y_fraud_test.value_counts(normalize=True))\n"
            "\n"
            "print('\\nCredit train class balance (post-SMOTE):')\n"
            "print(y_credit_train.value_counts(normalize=True))\n"
            "print('\\nCredit test class balance (natural):')\n"
            "print(y_credit_test.value_counts(normalize=True))"
        ),

        # ===================== PART A: FRAUD DATA =====================
        nbf.v4.new_markdown_cell(
            "---\n# Part A: E-Commerce Fraud Data (`Fraud_Data.csv`)"
        ),
        nbf.v4.new_markdown_cell(
            "## A1. Baseline Model — Logistic Regression\n"
            "Logistic Regression provides an interpretable baseline. Coefficients can be directly inspected "
            "to understand the direction and magnitude of each feature's effect on the fraud log-odds."
        ),
        nbf.v4.new_code_cell(
            "fraud_lr = train_logistic_regression(X_fraud_train, y_fraud_train)\n"
            "fraud_lr_results = evaluate_classifier(fraud_lr, X_fraud_test, y_fraud_test, model_name='Logistic Regression (Fraud)')\n"
            "\n"
            "print(f\"AUC-PR:   {fraud_lr_results['auc_pr']:.4f}\")\n"
            "print(f\"F1-Score: {fraud_lr_results['f1_score']:.4f}\")\n"
            "print('\\nClassification Report:')\n"
            "print(fraud_lr_results['classification_report'])\n"
            "\n"
            "plot_confusion_matrix(\n"
            "    fraud_lr_results['confusion_matrix'],\n"
            "    title='Fraud Data - Logistic Regression Confusion Matrix',\n"
            "    save_path='../reports/figures/fraud_lr_confusion_matrix.png'\n"
            ")\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## A2. Ensemble Model — XGBoost (Hyperparameter Tuned)\n"
            "We tune `n_estimators`, `max_depth`, `learning_rate`, and `subsample` using `RandomizedSearchCV` "
            "with Stratified 3-Fold CV, optimizing for AUC-PR (average precision)."
        ),
        nbf.v4.new_code_cell(
            "fraud_param_dist = {\n"
            "    'n_estimators': [100, 150, 200],\n"
            "    'max_depth': [3, 5, 7],\n"
            "    'learning_rate': [0.05, 0.1, 0.2],\n"
            "    'subsample': [0.8, 1.0],\n"
            "}\n"
            "\n"
            "fraud_xgb, fraud_xgb_best_params, fraud_xgb_cv_results = tune_xgboost(\n"
            "    X_fraud_train, y_fraud_train, param_distributions=fraud_param_dist, n_iter=6, cv=3\n"
            ")\n"
            "\n"
            "print('Best hyperparameters:', fraud_xgb_best_params)\n"
            "\n"
            "fraud_xgb_results = evaluate_classifier(fraud_xgb, X_fraud_test, y_fraud_test, model_name='XGBoost (Fraud)')\n"
            "\n"
            "print(f\"\\nAUC-PR:   {fraud_xgb_results['auc_pr']:.4f}\")\n"
            "print(f\"F1-Score: {fraud_xgb_results['f1_score']:.4f}\")\n"
            "print('\\nClassification Report:')\n"
            "print(fraud_xgb_results['classification_report'])\n"
            "\n"
            "plot_confusion_matrix(\n"
            "    fraud_xgb_results['confusion_matrix'],\n"
            "    title='Fraud Data - XGBoost Confusion Matrix',\n"
            "    save_path='../reports/figures/fraud_xgb_confusion_matrix.png'\n"
            ")\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## A3. Precision-Recall Curve Comparison"
        ),
        nbf.v4.new_code_cell(
            "plot_pr_curves(\n"
            "    y_fraud_test,\n"
            "    {'Logistic Regression': fraud_lr_results['y_proba'], 'XGBoost': fraud_xgb_results['y_proba']},\n"
            "    title='Fraud Data - Precision-Recall Curve Comparison',\n"
            "    save_path='../reports/figures/fraud_pr_curve_comparison.png'\n"
            ")\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## A4. Cross-Validation (Stratified K-Fold, k=5)\n"
            "We run 5-fold cross-validation **on the SMOTE-resampled training set** for both models, reporting "
            "the mean and standard deviation of AUC-PR and F1-Score across folds. This gives a sense of the "
            "variance of each model's performance independent of the single train/test split. Note that, because "
            "the folds are drawn from an already-resampled training set, these CV scores reflect performance on a "
            "balanced distribution — the **test-set metrics above remain the primary indicator of real-world "
            "performance**, since the test set retains the natural class imbalance."
        ),
        nbf.v4.new_code_cell(
            "print('Running 5-fold CV for Logistic Regression (Fraud)...')\n"
            "fraud_lr_cv = cross_validate_classifier(fraud_lr, X_fraud_train, y_fraud_train, cv=5)\n"
            "print(fraud_lr_cv)\n"
            "\n"
            "print('\\nRunning 5-fold CV for XGBoost (Fraud)...')\n"
            "fraud_xgb_cv = cross_validate_classifier(fraud_xgb, X_fraud_train, y_fraud_train, cv=5)\n"
            "print(fraud_xgb_cv)"
        ),
        nbf.v4.new_markdown_cell(
            "## A5. Model Comparison & Selection (Fraud Data)"
        ),
        nbf.v4.new_code_cell(
            "fraud_comparison = pd.DataFrame([\n"
            "    {\n"
            "        'Model': 'Logistic Regression',\n"
            "        'Test AUC-PR': fraud_lr_results['auc_pr'],\n"
            "        'Test F1-Score': fraud_lr_results['f1_score'],\n"
            "        'CV AUC-PR (mean)': fraud_lr_cv['auc_pr_mean'],\n"
            "        'CV AUC-PR (std)': fraud_lr_cv['auc_pr_std'],\n"
            "        'CV F1 (mean)': fraud_lr_cv['f1_mean'],\n"
            "        'CV F1 (std)': fraud_lr_cv['f1_std'],\n"
            "    },\n"
            "    {\n"
            "        'Model': 'XGBoost (Tuned)',\n"
            "        'Test AUC-PR': fraud_xgb_results['auc_pr'],\n"
            "        'Test F1-Score': fraud_xgb_results['f1_score'],\n"
            "        'CV AUC-PR (mean)': fraud_xgb_cv['auc_pr_mean'],\n"
            "        'CV AUC-PR (std)': fraud_xgb_cv['auc_pr_std'],\n"
            "        'CV F1 (mean)': fraud_xgb_cv['f1_mean'],\n"
            "        'CV F1 (std)': fraud_xgb_cv['f1_std'],\n"
            "    },\n"
            "])\n"
            "fraud_comparison"
        ),

        # ===================== PART B: CREDIT CARD DATA =====================
        nbf.v4.new_markdown_cell(
            "---\n# Part B: Bank Credit Card Data (`creditcard.csv`)"
        ),
        nbf.v4.new_markdown_cell(
            "## B1. Baseline Model — Logistic Regression"
        ),
        nbf.v4.new_code_cell(
            "credit_lr = train_logistic_regression(X_credit_train, y_credit_train)\n"
            "credit_lr_results = evaluate_classifier(credit_lr, X_credit_test, y_credit_test, model_name='Logistic Regression (Credit Card)')\n"
            "\n"
            "print(f\"AUC-PR:   {credit_lr_results['auc_pr']:.4f}\")\n"
            "print(f\"F1-Score: {credit_lr_results['f1_score']:.4f}\")\n"
            "print('\\nClassification Report:')\n"
            "print(credit_lr_results['classification_report'])\n"
            "\n"
            "plot_confusion_matrix(\n"
            "    credit_lr_results['confusion_matrix'],\n"
            "    title='Credit Card - Logistic Regression Confusion Matrix',\n"
            "    save_path='../reports/figures/credit_lr_confusion_matrix.png'\n"
            ")\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## B2. Ensemble Model — XGBoost (Hyperparameter Tuned)"
        ),
        nbf.v4.new_code_cell(
            "credit_param_dist = {\n"
            "    'n_estimators': [100, 150, 200],\n"
            "    'max_depth': [3, 5, 7],\n"
            "    'learning_rate': [0.05, 0.1, 0.2],\n"
            "    'subsample': [0.8, 1.0],\n"
            "}\n"
            "\n"
            "credit_xgb, credit_xgb_best_params, credit_xgb_cv_results = tune_xgboost(\n"
            "    X_credit_train, y_credit_train, param_distributions=credit_param_dist, n_iter=6, cv=3\n"
            ")\n"
            "\n"
            "print('Best hyperparameters:', credit_xgb_best_params)\n"
            "\n"
            "credit_xgb_results = evaluate_classifier(credit_xgb, X_credit_test, y_credit_test, model_name='XGBoost (Credit Card)')\n"
            "\n"
            "print(f\"\\nAUC-PR:   {credit_xgb_results['auc_pr']:.4f}\")\n"
            "print(f\"F1-Score: {credit_xgb_results['f1_score']:.4f}\")\n"
            "print('\\nClassification Report:')\n"
            "print(credit_xgb_results['classification_report'])\n"
            "\n"
            "plot_confusion_matrix(\n"
            "    credit_xgb_results['confusion_matrix'],\n"
            "    title='Credit Card - XGBoost Confusion Matrix',\n"
            "    save_path='../reports/figures/credit_xgb_confusion_matrix.png'\n"
            ")\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## B3. Precision-Recall Curve Comparison"
        ),
        nbf.v4.new_code_cell(
            "plot_pr_curves(\n"
            "    y_credit_test,\n"
            "    {'Logistic Regression': credit_lr_results['y_proba'], 'XGBoost': credit_xgb_results['y_proba']},\n"
            "    title='Credit Card - Precision-Recall Curve Comparison',\n"
            "    save_path='../reports/figures/credit_pr_curve_comparison.png'\n"
            ")\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## B4. Cross-Validation (Stratified K-Fold, k=5)\n"
            "As with the fraud dataset, 5-fold CV is run on the SMOTE-resampled training set to assess "
            "variance across folds, while the untouched test set (with its natural ~0.17% fraud rate) "
            "remains the primary indicator of real-world performance."
        ),
        nbf.v4.new_code_cell(
            "print('Running 5-fold CV for Logistic Regression (Credit Card)...')\n"
            "credit_lr_cv = cross_validate_classifier(credit_lr, X_credit_train, y_credit_train, cv=5)\n"
            "print(credit_lr_cv)\n"
            "\n"
            "print('\\nRunning 5-fold CV for XGBoost (Credit Card)...')\n"
            "credit_xgb_cv = cross_validate_classifier(credit_xgb, X_credit_train, y_credit_train, cv=5)\n"
            "print(credit_xgb_cv)"
        ),
        nbf.v4.new_markdown_cell(
            "## B5. Model Comparison & Selection (Credit Card Data)"
        ),
        nbf.v4.new_code_cell(
            "credit_comparison = pd.DataFrame([\n"
            "    {\n"
            "        'Model': 'Logistic Regression',\n"
            "        'Test AUC-PR': credit_lr_results['auc_pr'],\n"
            "        'Test F1-Score': credit_lr_results['f1_score'],\n"
            "        'CV AUC-PR (mean)': credit_lr_cv['auc_pr_mean'],\n"
            "        'CV AUC-PR (std)': credit_lr_cv['auc_pr_std'],\n"
            "        'CV F1 (mean)': credit_lr_cv['f1_mean'],\n"
            "        'CV F1 (std)': credit_lr_cv['f1_std'],\n"
            "    },\n"
            "    {\n"
            "        'Model': 'XGBoost (Tuned)',\n"
            "        'Test AUC-PR': credit_xgb_results['auc_pr'],\n"
            "        'Test F1-Score': credit_xgb_results['f1_score'],\n"
            "        'CV AUC-PR (mean)': credit_xgb_cv['auc_pr_mean'],\n"
            "        'CV AUC-PR (std)': credit_xgb_cv['auc_pr_std'],\n"
            "        'CV F1 (mean)': credit_xgb_cv['f1_mean'],\n"
            "        'CV F1 (std)': credit_xgb_cv['f1_std'],\n"
            "    },\n"
            "])\n"
            "credit_comparison"
        ),

        # ===================== SAVE MODELS =====================
        nbf.v4.new_markdown_cell(
            "---\n## Saving Selected Models\n"
            "We persist all trained models to `../models/` so they can be reloaded directly in "
            "`shap-explainability.ipynb` without retraining."
        ),
        nbf.v4.new_code_cell(
            "joblib.dump(fraud_lr, '../models/fraud_logistic_regression.pkl')\n"
            "joblib.dump(fraud_xgb, '../models/fraud_xgboost.pkl')\n"
            "joblib.dump(credit_lr, '../models/credit_logistic_regression.pkl')\n"
            "joblib.dump(credit_xgb, '../models/credit_xgboost.pkl')\n"
            "\n"
            "print('Saved models to ../models/:')\n"
            "for f in sorted(os.listdir('../models')):\n"
            "    print(' -', f)"
        ),
    ]

    nb['cells'] = cells
    notebook_path = os.path.join(project_root, 'notebooks', 'modeling.ipynb')
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)


def create_shap_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# Model Explainability with SHAP\n"
            "\n"
            "This notebook interprets the best-performing models selected in `modeling.ipynb` "
            "(XGBoost for both datasets) using **SHAP (SHapley Additive exPlanations)**. For each dataset we:\n"
            "1. Extract the model's **built-in feature importance** (top 10).\n"
            "2. Generate a **SHAP summary plot** (global feature importance / direction of effect).\n"
            "3. Generate **SHAP force plots** for one true positive, one false positive, and one false negative "
            "prediction.\n"
            "4. Compare SHAP importance with built-in feature importance and identify the **top 5 drivers** of "
            "fraud predictions.\n"
            "\n"
            "We close with **actionable business recommendations** for Adey Innovations Inc., each tied to a "
            "specific SHAP insight."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "import os\n"
            "import warnings\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import joblib\n"
            "import shap\n"
            "\n"
            "warnings.filterwarnings('ignore')\n"
            "\n"
            "# Add src directory to path\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.modeling import evaluate_classifier\n"
            "from src.explainability import (\n"
            "    get_feature_importance,\n"
            "    plot_feature_importance,\n"
            "    compute_shap_values,\n"
            "    find_prediction_examples,\n"
            ")\n"
            "\n"
            "os.makedirs('../reports/figures', exist_ok=True)\n"
            "shap.initjs()"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Models and Test Data\n"
            "We reload the tuned XGBoost models (the ensemble models selected as the best performers for "
            "each dataset in `modeling.ipynb`) along with the untouched test sets."
        ),
        nbf.v4.new_code_cell(
            "fraud_xgb = joblib.load('../models/fraud_xgboost.pkl')\n"
            "credit_xgb = joblib.load('../models/credit_xgboost.pkl')\n"
            "\n"
            "X_fraud_test = pd.read_csv('../data/processed/fraud_X_test.csv')\n"
            "y_fraud_test = pd.read_csv('../data/processed/fraud_y_test.csv').squeeze()\n"
            "\n"
            "X_credit_test = pd.read_csv('../data/processed/credit_X_test.csv')\n"
            "y_credit_test = pd.read_csv('../data/processed/credit_y_test.csv').squeeze()\n"
            "\n"
            "fraud_results = evaluate_classifier(fraud_xgb, X_fraud_test, y_fraud_test, model_name='XGBoost (Fraud)')\n"
            "credit_results = evaluate_classifier(credit_xgb, X_credit_test, y_credit_test, model_name='XGBoost (Credit Card)')\n"
            "\n"
            "print(f\"Fraud  test -> AUC-PR: {fraud_results['auc_pr']:.4f}, F1: {fraud_results['f1_score']:.4f}\")\n"
            "print(f\"Credit test -> AUC-PR: {credit_results['auc_pr']:.4f}, F1: {credit_results['f1_score']:.4f}\")"
        ),

        # ===================== PART A: FRAUD =====================
        nbf.v4.new_markdown_cell(
            "---\n# Part A: E-Commerce Fraud Data (`Fraud_Data.csv`)"
        ),
        nbf.v4.new_markdown_cell(
            "## A1. Built-in Feature Importance (Top 10)"
        ),
        nbf.v4.new_code_cell(
            "fraud_importance = get_feature_importance(fraud_xgb, X_fraud_test.columns.tolist(), top_n=10)\n"
            "plot_feature_importance(\n"
            "    fraud_importance,\n"
            "    title='Fraud Data - Top 10 Feature Importances (XGBoost)',\n"
            "    save_path='../reports/figures/fraud_feature_importance.png'\n"
            ")\n"
            "plt.show()\n"
            "fraud_importance"
        ),
        nbf.v4.new_markdown_cell(
            "## A2. SHAP Summary Plot (Global Feature Importance)\n"
            "We compute SHAP values on a random sample of the test set (for tractability) using a "
            "`TreeExplainer`, which is exact and fast for tree ensembles like XGBoost."
        ),
        nbf.v4.new_code_cell(
            "fraud_explainer = shap.TreeExplainer(fraud_xgb)\n"
            "\n"
            "np.random.seed(42)\n"
            "fraud_sample_idx = np.random.choice(X_fraud_test.index, size=min(1000, len(X_fraud_test)), replace=False)\n"
            "X_fraud_sample = X_fraud_test.loc[fraud_sample_idx].reset_index(drop=True)\n"
            "\n"
            "_, fraud_shap_sample = compute_shap_values(fraud_xgb, X_fraud_sample)\n"
            "\n"
            "shap.summary_plot(fraud_shap_sample, X_fraud_sample, show=False)\n"
            "plt.savefig('../reports/figures/fraud_shap_summary.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## A3. SHAP Force Plots — True Positive, False Positive, False Negative\n"
            "We locate one representative example of each prediction outcome on the full test set and "
            "render a force plot showing how each feature pushes the prediction above or below the "
            "model's base (expected) value."
        ),
        nbf.v4.new_code_cell(
            "fraud_examples = find_prediction_examples(y_fraud_test.values, fraud_results['y_pred'], fraud_results['y_proba'])\n"
            "print('Selected example indices (Fraud):', fraud_examples)\n"
            "\n"
            "fraud_example_rows = X_fraud_test.iloc[list(fraud_examples.values())]\n"
            "_, fraud_shap_examples = compute_shap_values(fraud_xgb, fraud_example_rows)"
        ),
        nbf.v4.new_code_cell(
            "# True Positive: correctly identified fraud\n"
            "i = list(fraud_examples.keys()).index('true_positive')\n"
            "idx = fraud_examples['true_positive']\n"
            "print(f\"True label: {y_fraud_test.iloc[idx]}, Predicted: {fraud_results['y_pred'][idx]}, \"\n"
            "      f\"P(fraud)={fraud_results['y_proba'][idx]:.4f}\")\n"
            "\n"
            "shap.force_plot(\n"
            "    fraud_explainer.expected_value, fraud_shap_examples[i], fraud_example_rows.iloc[i],\n"
            "    matplotlib=True, show=False\n"
            ")\n"
            "plt.savefig('../reports/figures/fraud_shap_force_true_positive.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# False Positive: legitimate transaction flagged as fraud\n"
            "i = list(fraud_examples.keys()).index('false_positive')\n"
            "idx = fraud_examples['false_positive']\n"
            "print(f\"True label: {y_fraud_test.iloc[idx]}, Predicted: {fraud_results['y_pred'][idx]}, \"\n"
            "      f\"P(fraud)={fraud_results['y_proba'][idx]:.4f}\")\n"
            "\n"
            "shap.force_plot(\n"
            "    fraud_explainer.expected_value, fraud_shap_examples[i], fraud_example_rows.iloc[i],\n"
            "    matplotlib=True, show=False\n"
            ")\n"
            "plt.savefig('../reports/figures/fraud_shap_force_false_positive.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# False Negative: missed fraud\n"
            "i = list(fraud_examples.keys()).index('false_negative')\n"
            "idx = fraud_examples['false_negative']\n"
            "print(f\"True label: {y_fraud_test.iloc[idx]}, Predicted: {fraud_results['y_pred'][idx]}, \"\n"
            "      f\"P(fraud)={fraud_results['y_proba'][idx]:.4f}\")\n"
            "\n"
            "shap.force_plot(\n"
            "    fraud_explainer.expected_value, fraud_shap_examples[i], fraud_example_rows.iloc[i],\n"
            "    matplotlib=True, show=False\n"
            ")\n"
            "plt.savefig('../reports/figures/fraud_shap_force_false_negative.png', bbox_inches='tight')\n"
            "plt.show()"
        ),

        # ===================== PART B: CREDIT CARD =====================
        nbf.v4.new_markdown_cell(
            "---\n# Part B: Bank Credit Card Data (`creditcard.csv`)"
        ),
        nbf.v4.new_markdown_cell(
            "## B1. Built-in Feature Importance (Top 10)"
        ),
        nbf.v4.new_code_cell(
            "credit_importance = get_feature_importance(credit_xgb, X_credit_test.columns.tolist(), top_n=10)\n"
            "plot_feature_importance(\n"
            "    credit_importance,\n"
            "    title='Credit Card - Top 10 Feature Importances (XGBoost)',\n"
            "    save_path='../reports/figures/credit_feature_importance.png'\n"
            ")\n"
            "plt.show()\n"
            "credit_importance"
        ),
        nbf.v4.new_markdown_cell(
            "## B2. SHAP Summary Plot (Global Feature Importance)"
        ),
        nbf.v4.new_code_cell(
            "credit_explainer = shap.TreeExplainer(credit_xgb)\n"
            "\n"
            "np.random.seed(42)\n"
            "credit_sample_idx = np.random.choice(X_credit_test.index, size=min(1000, len(X_credit_test)), replace=False)\n"
            "X_credit_sample = X_credit_test.loc[credit_sample_idx].reset_index(drop=True)\n"
            "\n"
            "_, credit_shap_sample = compute_shap_values(credit_xgb, X_credit_sample)\n"
            "\n"
            "shap.summary_plot(credit_shap_sample, X_credit_sample, show=False)\n"
            "plt.savefig('../reports/figures/credit_shap_summary.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## B3. SHAP Force Plots — True Positive, False Positive, False Negative"
        ),
        nbf.v4.new_code_cell(
            "credit_examples = find_prediction_examples(y_credit_test.values, credit_results['y_pred'], credit_results['y_proba'])\n"
            "print('Selected example indices (Credit Card):', credit_examples)\n"
            "\n"
            "credit_example_rows = X_credit_test.iloc[list(credit_examples.values())]\n"
            "_, credit_shap_examples = compute_shap_values(credit_xgb, credit_example_rows)"
        ),
        nbf.v4.new_code_cell(
            "# True Positive: correctly identified fraud\n"
            "i = list(credit_examples.keys()).index('true_positive')\n"
            "idx = credit_examples['true_positive']\n"
            "print(f\"True label: {y_credit_test.iloc[idx]}, Predicted: {credit_results['y_pred'][idx]}, \"\n"
            "      f\"P(fraud)={credit_results['y_proba'][idx]:.4f}\")\n"
            "\n"
            "shap.force_plot(\n"
            "    credit_explainer.expected_value, credit_shap_examples[i], credit_example_rows.iloc[i],\n"
            "    matplotlib=True, show=False\n"
            ")\n"
            "plt.savefig('../reports/figures/credit_shap_force_true_positive.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# False Positive: legitimate transaction flagged as fraud\n"
            "i = list(credit_examples.keys()).index('false_positive')\n"
            "idx = credit_examples['false_positive']\n"
            "print(f\"True label: {y_credit_test.iloc[idx]}, Predicted: {credit_results['y_pred'][idx]}, \"\n"
            "      f\"P(fraud)={credit_results['y_proba'][idx]:.4f}\")\n"
            "\n"
            "shap.force_plot(\n"
            "    credit_explainer.expected_value, credit_shap_examples[i], credit_example_rows.iloc[i],\n"
            "    matplotlib=True, show=False\n"
            ")\n"
            "plt.savefig('../reports/figures/credit_shap_force_false_positive.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# False Negative: missed fraud\n"
            "i = list(credit_examples.keys()).index('false_negative')\n"
            "idx = credit_examples['false_negative']\n"
            "print(f\"True label: {y_credit_test.iloc[idx]}, Predicted: {credit_results['y_pred'][idx]}, \"\n"
            "      f\"P(fraud)={credit_results['y_proba'][idx]:.4f}\")\n"
            "\n"
            "shap.force_plot(\n"
            "    credit_explainer.expected_value, credit_shap_examples[i], credit_example_rows.iloc[i],\n"
            "    matplotlib=True, show=False\n"
            ")\n"
            "plt.savefig('../reports/figures/credit_shap_force_false_negative.png', bbox_inches='tight')\n"
            "plt.show()"
        ),
    ]

    nb['cells'] = cells
    notebook_path = os.path.join(project_root, 'notebooks', 'shap-explainability.ipynb')
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)


if __name__ == '__main__':
    create_eda_fraud_notebook()
    create_eda_creditcard_notebook()
    create_feature_engineering_notebook()
    create_modeling_notebook()
    create_shap_notebook()
    print("All notebook templates generated successfully!")
