from fastapi.testclient import TestClient
from app.main import app

def test_model_info():
    with TestClient(app) as client:
        r = client.get("/v1/model-info")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert "has_predict_proba" in data
