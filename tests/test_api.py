"""Testes da API FastAPI — health, predict sem modelo e predict com mock."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import customer_churn_ibm.api.main as api_module
from customer_churn_ibm.api.main import app

_EXAMPLE_PAYLOAD = {
    "tenure_months": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_without_model_returns_503(client):
    """Sem modelo carregado (estado padrão nos testes), /predict retorna 503."""
    api_module._model = None
    response = client.post("/predict", json=_EXAMPLE_PAYLOAD)
    assert response.status_code == 503


def test_predict_with_mocked_model_returns_valid_response(client):
    """Com modelo mockado, /predict retorna churn_probability no intervalo [0, 1]."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    api_module._model = mock_model
    try:
        response = client.post("/predict", json=_EXAMPLE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert "churn_probability" in body
        assert "churn_label" in body
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert isinstance(body["churn_label"], bool)
    finally:
        api_module._model = None


def test_predict_payload_schema_validation(client):
    """Payload inválido (campo obrigatório ausente) deve retornar 422."""
    incomplete = {k: v for k, v in _EXAMPLE_PAYLOAD.items() if k != "tenure_months"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
