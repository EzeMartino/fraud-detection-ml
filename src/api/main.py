from contextlib import asynccontextmanager
from functools import lru_cache
import json

from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, ConfigDict
from typing import Optional

from src.config import get_active_metadata_path, get_active_pipeline_path
from src.models.predict import predict_single


class FraudInput(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    threshold: Optional[float] = 0.5

    model_config = ConfigDict(extra="forbid")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.pipeline = get_pipeline()
    app.state.metadata = get_metadata()
    yield
    # Shutdown (opcional)

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

@lru_cache(maxsize=1)
def get_pipeline():
    pipeline_path = get_active_pipeline_path()
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {pipeline_path}. "
            "Run 'train_random_forest.py' to generate model artifacts first"
        )
    return joblib.load(pipeline_path)

@lru_cache(maxsize=1)
def get_metadata():
    metadata_path = get_active_metadata_path()
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {metadata_path}. "
            "Run 'train_random_forest.py' to generate model artifacts first"
        )
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model_loaded": True,
        "model_version": app.state.metadata["model_version"]
        }

@app.post("/predict")
def predict(payload: FraudInput):
    try:
        data = payload.model_dump()
        threshold = data.pop("threshold", 0.5)
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        
        pipeline = app.state.pipeline
        metadata = app.state.metadata
        
        result = predict_single(
            data=data, 
            pipeline=pipeline, 
            metadata=metadata, 
            threshold=threshold
            )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))