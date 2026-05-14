"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import pytest
from litestar.testing import TestClient

from app.model_utils import predict_churn
from main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FEATURES = [600, "France", "Male", 40, 3, 60000.0, 2, 1, 1, 50000.0]

SAMPLE_PAYLOAD = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0,
}


# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------

def test_predict_churn_returns_binary():
    result = predict_churn(SAMPLE_FEATURES)
    assert result in (0, 1)


@pytest.mark.parametrize("credit_score, balance", [
    (850, 0.0),       # max credit score, zero balance
    (300, 250000.0),  # min credit score, high balance
])
def test_predict_churn_edge_cases(credit_score, balance):
    features = [credit_score, "Germany", "Female", 60, 10, balance, 1, 0, 0, 200000.0]
    result = predict_churn(features)
    assert result in (0, 1)


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

def test_predict_endpoint():
    with TestClient(app=app) as client:
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 201
    assert "churn_prediction" in response.json()
    assert response.json()["churn_prediction"] in (0, 1)


def test_health_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_home_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@pytest.mark.parametrize("bad_payload", [
    {**SAMPLE_PAYLOAD, "CreditScore": "not-a-number"},  # wrong type
    {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "Age"},  # missing field
    {},  # empty body
])
def test_predict_endpoint_invalid_input(bad_payload):
    with TestClient(app=app) as client:
        response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400