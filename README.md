# fraud-detection-ml

End-to-end machine learning system for fraud detection in highly imbalanced data (~0.17% fraud rate), optimized for operational decision-making using top-K precision instead of global metrics.

**Key result:**
The model is optimized to maximize fraud detection within a fixed investigation capacity using top-K precision.

- Model: Calibrated Random Forest
- Primary metric: Precision@K
- Model Selection Metric: precision_at_0_2pct
- Secondary metric: PR-AUC
- Business focus: maximize fraud captured in top-ranked transactions


## Business Problem

In fraud detection, reviewing every transaction is impossible.

The goal is to rank transactions by risk so that:
- The top X% can be reviewed by analysts
- Fraud is captured efficiently with limited resources

This project focuses on optimizing performance at the top of the ranking (Top-K), not overall accuracy.


## Dataset

**Credit Card Fraud Detection Dataset (Kaggle)**
- Transactions: 284,807
- Fraud cases: 492 (~0.17%)

This dataset contains anonymized credit card transactions and an extremely imbalanced binary target:

- `Class = 1` → fraud
- `Class = 0` → non-fraud

Main challenge:
- extreme class imbalance
- precision vs recall trade-off
- operational threshold definition
- features V1–V28 are PCA-transformed → no business interpretability

Note: Time and Amount are the only non-transformed features.


## Approach

Pipeline:
Raw Data
↓
Preprocessing
↓
Model Training (Logistic, Random Forest)
↓
Calibration (Isotonic)
↓
Top-K / Threshold Analysis
↓
Model Selection
↓
Artifact Export
↓
Inference API


## Model Selection

Final model: Calibrated Random Forest

Why:
- Significantly better performance in top-K segments (critical for fraud operations)
- Calibration improved probability reliability for threshold-based decisions
- Trade-off between performance and computational cost was acceptable

Key decision:
Model selection was driven by operational performance (top-K), not global metrics.

Trade-off:
Higher precision at top-K was prioritized over global recall to match operational constraints.


## Evaluation

Primary metric:
- Precision@K (Top 1%, 0.5%, 0.2%)

Secondary metrics:
- PR-AUC (handles class imbalance)
- ROC-AUC (ranking only)

Key insight:
ROC-AUC alone is insufficient for operational decisions in imbalanced datasets.
In extremely imbalanced scenarios, a model can achieve high ROC-AUC while still generating unusable alert volumes due to low precision.


## Results

| Metric        | Value |
|--------------|------|
| ROC-AUC      | 0.97 |
| PR-AUC       | 0.83 |
| Precision@1% | ~14.29%  |
| Recall@1%    | ~85.26%  |
| Precision@0.5% | ~27.91%  |
| Recall@0.5%    | ~83.16%  |
| Precision@0.2% | ~69.03%  |
| Recall@0.2%    | ~82.11%  |

Results demonstrate that fraud is highly concentrated in the top-ranked transactions, enabling efficient allocation of investigation resources.


## Calibration

Random Forest probabilities were poorly calibrated.

Isotonic calibration was applied to:
- Improve probability reliability
- Enable better threshold selection

This is critical in risk-based decision systems because uncalibrated probabilities led to unreliable threshold decisions.


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


## Setup & Reproducibility

### Clone Repo

### Create virtual environment
Windows:
```bash
python -m venv .venv
source .venv/Scripts/activate
```
Linux/Mac:
```bash
source .venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run data profiling
```bash
python -m src.data.load_and_profile
```
### Get best config:
```bash
python -m src.models.tune_random_forest
```
### Train
```bash
python -m src.models.train_random_forest
```
### Predict:
```bash
python src/predict.py
```
### Run API:
```bash
uvicorn src.api.main:app --reload
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
    "V28":2
}
```
Output:
```json
{
    "fraud_score": 0.009379329174404236, 
    "predicted_class": 0, 
    "threshold": 0.5,
    "model_version": "2d0b2262b407"
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
- No concept drift handling


## Future Work
- Real feature engineering with temporal data
- Online inference pipeline
- Monitoring & drift detection


## Status
Project is production-ready at prototype level:
- Reproducible training pipeline
- Exported artifacts
- Inference script
- API with validation
- Automated tests