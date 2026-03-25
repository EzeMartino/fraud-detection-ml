from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API is running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        
    data = response.json()
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in data
    assert "model_version" in data
    assert data["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "Time": 0,
        "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "threshold": 0.5,
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "fraud_score" in body
    assert "predicted_class" in body
    assert "threshold" in body