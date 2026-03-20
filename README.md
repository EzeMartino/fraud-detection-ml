# fraud-detection-ml

End-to-end machine learning project for credit card fraud detection, focused on highly imbalanced classification, threshold optimization, explainability, and reproducible ML workflows.


## Project Goal

Build a portfolio-ready fraud detection system that can:

- profile and validate raw transaction data
- train baseline and tree-based models
- optimize decision thresholds under class imbalance
- export reproducible artifacts
- support batch inference and API deployment


## Dataset

**Credit Card Fraud Detection Dataset**

This dataset contains anonymized credit card transactions and an extremely imbalanced binary target:

- `Class = 1` → fraud
- `Class = 0` → non-fraud

Main challenge:
- extreme class imbalance
- precision vs recall trade-off
- operational threshold definition


## Project Structure

```text
data/          raw and processed datasets
notebooks/     exploratory analysis
reports/       profiling outputs and evaluation reports
artifacts/     exported metadata and threshold configs
models/        trained model files
src/           source code
tests/         automated tests
sql/           complementary SQL practice
```


## Initial Scope

Phase 1 of this project focuses on:
- dataset setup
- data profiling
- reproducible baseline training
- evaluation under severe class imbalance


## Metrics

Primary evaluation will emphasize:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- Threshold-based operational analysis

Accuracy will be treated as a secondary reference only, since it is misleading under extreme imbalance.


## Setup

### Create virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run data profiling
```bash
python -m src.data.load_and_profile
```
### Run tests
```bash
pytest
```

## 
## Inference

To run a sample prediction:

```bash
python -m src.models.predict
```
Example input:
```json
{
    "Time": 3500,
    "Amount": 100.0,
    "amount_log":10,
    "hour":23,
    "is_night":True,
    "amount_zscore":5,
    "V1":2,
    "V2":2,
    "V3":2,
    "V4":2,
    "V5":2,
    "V6":2,
    "V7":2,
    "V8":2,
    "V9":2,
    "V10":2,
    "V11":2,
    "V12":2,
    "V13":2,
    "V14":2,
    "V15":2,
    "V16":2,
    "V17":2,
    "V18":2,
    "V19":2,
    "V20":2,
    "V21":2,
    "V22":2,
    "V23":2,
    "V24":2,
    "V25":2,
    "V26":2,
    "V27":2,
    "V28":2,
}
```
Output:
```json
{
    'fraud_score': 0.009379329174404236, 
    'predicted_class': 0, 
    'threshold': 0.5
}
```
## Roadmap

Week 5: dataset setup + baseline

Week 6: feature engineering

Week 7: final model + explainability

Week 8: packaging + deployment

## Status

Project initialization in progress.