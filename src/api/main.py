from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional

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


app = FastAPI(title="Fraud Detection API")


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: FraudInput):
    try:
        data = payload.model_dump()
        threshold = data.pop("threshold", 0.5)
        result = predict_single(data, threshold=threshold)
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))