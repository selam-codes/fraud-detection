# Scripts

This directory contains helper scripts for the project:

- **`generate_notebooks.py`**:
  - Automatically generates and populates the Jupyter notebooks (`eda-fraud-data.ipynb`, `eda-creditcard.ipynb`, and `feature-engineering.ipynb`) with complete pipelines, descriptions, and code blocks.
  - Resolves pathing relative to the project root directory.

## How to Run

To regenerate the Jupyter notebook templates:
```bash
python3 scripts/generate_notebooks.py
```
