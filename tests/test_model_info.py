from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_model_info():
    r = client.get("/v1/model-info")
    assert r.status_code == 200
    data = r.json()
    # jeśli metadata.json istnieje, powinny być np. class_names
    assert isinstance(data, dict)
