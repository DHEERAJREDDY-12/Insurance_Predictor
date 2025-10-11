# Title Insurance Cost Prediction

Predict title insurance premium using machine learning.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure Kaggle API (optional, if downloading from Kaggle):
   - Create a Kaggle account and an API Token from Account > API > Create New Token.
   - Place `kaggle.json` at:
     - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Or set env vars:
```bash
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
```

## Data

- Option A: Download from Kaggle via the notebook or CLI.
- Option B: Use a local CSV with `--data-path` when training.

## Quickstart

Train a model:
```bash
python src/train.py --data-path data/train.csv --target premium --model-out artifacts/best_model.joblib
```

Run inference:
```bash
python src/infer.py --model artifacts/best_model.joblib --input-csv data/sample_predict.csv --output-csv data/predictions.csv
```

## Repository Structure

- `notebooks/` exploratory analysis and experimentation
- `src/train.py` training and model selection
- `src/infer.py` batch predictions
- `artifacts/` saved models and metrics
- `data/` datasets (not committed)
