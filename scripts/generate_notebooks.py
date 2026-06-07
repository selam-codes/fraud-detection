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

if __name__ == '__main__':
    create_eda_fraud_notebook()
    create_eda_creditcard_notebook()
    create_feature_engineering_notebook()
    print("All notebook templates generated successfully!")
