import pytest
from fastapi.testclient import TestClient
import json
import joblib
import pandas as pd
import numpy as np
from app.main import app, predictor_instance
from app.config import app_config
from transactionguard.features.engineering import FeatureEngineer
from sklearn.linear_model import LogisticRegression

@pytest.fixture(autouse=True)
def mock_artifacts(tmp_path):
    app_config.model_artifact_dir = tmp_path
    predictor_instance.artifact_dir = tmp_path
    model = LogisticRegression()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    model.fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    fe = FeatureEngineer()
    df_train = pd.DataFrame({'F1': [1.0, 3.0], 'F2': [2.0, 4.0], 'Class': [0, 1]})
    fe.fit(df_train)
    joblib.dump(fe, tmp_path / "feature_engineer.joblib")
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"model_name": "TestModel", "version": "v1.0", "threshold": 0.5, "feature_names": ["F1", "F2"]}, f)
    with open(tmp_path / "schema.json", "w") as f:
        json.dump({"F1": "float64", "F2": "float64"}, f)
    return tmp_path

def test_health_check_with_model():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200; assert response.json()["status"] == "ok"; assert response.json()["model_loaded"] is True

def test_readiness_check_with_model():
    with TestClient(app) as test_client:
        response = test_client.get("/ready")
        assert response.status_code == 200; assert response.json()["ready"] is True

def test_model_info():
    with TestClient(app) as test_client:
        response = test_client.get("/model-info")
        assert response.status_code == 200; data = response.json(); assert data["model_name"] == "TestModel"; assert data["version"] == "v1.0"; assert "F1" in data["features"]

def test_predict_single_valid():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5, "F2": 2.5})
        assert response.status_code == 200; data = response.json(); assert "probability" in data; assert "is_fraud" in data; assert data["threshold"] == 0.5

def test_predict_single_invalid():
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"F1": 1.5})
        assert response.status_code == 400; assert "Missing required features" in response.json()["detail"]

def test_predict_batch_valid():
    with TestClient(app) as test_client:
        response = test_client.post("/predict/batch", json=[{"F1": 1.5, "F2": 2.5}, {"F1": -1.5, "F2": -2.5}])
        assert response.status_code == 200; data = response.json(); assert len(data["predictions"]) == 2; assert data["summary"]["batch_size"] == 2

def test_validate_endpoint():
    with TestClient(app) as test_client:
        response = test_client.post("/validate", json={"F1": 1.5, "F2": 2.5})
        assert response.status_code == 200; assert response.json()["valid"] is True
