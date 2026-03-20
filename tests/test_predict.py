from src.models.predict import predict_single


def test_predict_single_runs():
    sample_input = {
        "Time": 0,
        "Amount": 100.0,
        "amount_log":10,
        "hour":23,
        "is_night":True,
        "amount_zscore":5,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

    result = predict_single(sample_input)

    assert "fraud_score" in result
    assert "predicted_class" in result
    assert 0.0 <= result["fraud_score"] <= 1.0