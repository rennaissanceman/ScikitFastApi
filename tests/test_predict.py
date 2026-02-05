from fastapi.testclient import TestClient
from app.main import app

def test_predict_ok():
    with TestClient(app) as client:
        r = client.post("/v1/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)

def test_predict_validation_fail():
    with TestClient(app) as client:
        r = client.post("/v1/predict", json={"features": [1.0, 2.0]})
        assert r.status_code == 422
