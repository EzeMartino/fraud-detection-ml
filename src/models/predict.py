import json

import joblib
import pandas as pd

from src.config import get_active_metadata_path, get_active_pipeline_path
    
def prepare_input(data: dict, features_used: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    missing_features = set(features_used) - set(df.columns)
    
    if missing_features:
        raise ValueError(f"Missing feature in input: {missing_features}")
    
    extra_features = set(df.columns) - set(features_used)
    if extra_features:
        raise ValueError(f"Unexpected features: {extra_features}")
    
    return df[features_used]

def predict_single(data: dict, metadata, pipeline, threshold: float = 0.5) -> dict:
    
    features_used = metadata["features_used"]
    X = prepare_input(data, features_used)
    
    fraud_score = float(pipeline.predict_proba(X)[:, 1][0])
    
    predicted_class = int(fraud_score >= threshold)
    
    return {
        "fraud_score": fraud_score,
        "predicted_class": predicted_class,
        "threshold": threshold,
        "model_version": metadata["model_version"]
    }
    
if __name__ == "__main__":
    sample_input = {
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

    metadata_path = get_active_metadata_path()
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {metadata_path}. "
            "Run 'train_random_forest.py' to generate model artifacts first"
        )
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    pipeline_path = get_active_pipeline_path()
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {pipeline_path}. "
            "Run 'train_random_forest.py' to generate model artifacts first"
        )
    pipeline = joblib.load(pipeline_path)
    
    result = predict_single(sample_input, metadata, pipeline)
    print(result)