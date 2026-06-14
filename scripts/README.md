# Scripts

This directory contains helper scripts for the project:

- **`generate_notebooks.py`**:
  - Automatically generates and populates the Jupyter notebooks (`eda-fraud-data.ipynb`, `eda-creditcard.ipynb`, `feature-engineering.ipynb`, `modeling.ipynb`, and `shap-explainability.ipynb`) with complete pipelines, descriptions, and code blocks.
  - Resolves pathing relative to the project root directory.
  - Note: `modeling.ipynb` and `shap-explainability.ipynb` were re-executed after generation (`jupyter nbconvert --execute`) and then annotated with model-selection and SHAP-interpretation markdown cells containing the actual computed results — regenerating from this script alone will reproduce the code/structure but not the narrative writeups with real numbers.

## How to Run

To regenerate the Jupyter notebook templates:
```bash
python3 scripts/generate_notebooks.py
```
