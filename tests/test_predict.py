import json

import joblib

from src.config import get_active_metadata_path, get_active_pipeline_path
from src.models.predict import predict_single


def test_predict_single_runs():
    sample_input = {
        "Time": 0,
        "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
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

    assert "fraud_score" in result
    assert "predicted_class" in result
    assert 0.0 <= result["fraud_score"] <= 1.0