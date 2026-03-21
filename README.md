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
- PR-AUC
- Precision@K
- Recall@K
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
    "fraud_score": 0.009379329174404236, 
    "predicted_class": 0, 
    "threshold": 0.5
}
```


## API
Run the API locally with:

```bash
uvicorn src.api.main:app --reload
```

Available endpoints:
- GET / → service message
- GET /health → health check
- POST /predict → fraud prediction

Interactive docs:
- http://127.0.0.1:8000/docs

Example request body:

```json
{
  "Time": 0,
  "Amount": 100.0,
  "V1": 0.0,
  "V2": 0.0,
  "V3": 0.0,
  "V4": 0.0,
  "V5": 0.0,
  "V6": 0.0,
  "V7": 0.0,
  "V8": 0.0,
  "V9": 0.0,
  "V10": 0.0,
  "V11": 0.0,
  "V12": 0.0,
  "V13": 0.0,
  "V14": 0.0,
  "V15": 0.0,
  "V16": 0.0,
  "V17": 0.0,
  "V18": 0.0,
  "V19": 0.0,
  "V20": 0.0,
  "V21": 0.0,
  "V22": 0.0,
  "V23": 0.0,
  "V24": 0.0,
  "V25": 0.0,
  "V26": 0.0,
  "V27": 0.0,
  "V28": 0.0,
  "threshold": 0.5
}
```


## Notes and Limitations

- Features `V1` to `V28` are PCA-transformed and anonymized, so domain-based feature engineering and business interpretability are limited.
- Additional engineered features based on `Amount` and `Time` were tested, but they did not improve top-k operational performance and were excluded from the final pipeline.
- Model selection was driven primarily by operational metrics such as precision@k and recall@k, not only by aggregate metrics.


## Roadmap

Week 5: dataset setup + baseline

Week 6: feature engineering

Week 7: final model + explainability

Week 8: packaging + deployment

## Status

## Status

Baseline modeling, threshold analysis, calibrated Random Forest selection, artifact export, local inference pipeline, and FastAPI service are complete.