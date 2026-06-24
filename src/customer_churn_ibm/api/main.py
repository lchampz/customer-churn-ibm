"""FastAPI — endpoints /health e /predict para inferência de churn."""

import logging
import pathlib
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from customer_churn_ibm.api.schemas import (ChurnRequest, ChurnResponse,
                                            HealthResponse)
from customer_churn_ibm.config import MODEL_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
)

# Mapeamento: campo API → coluna do DataFrame de treino
_FIELD_TO_COL: dict[str, str] = {
    "tenure_months": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
}

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model
    model_path = pathlib.Path(MODEL_DIR) / "mlp_pipeline.pkl"
    if model_path.exists():
        _model = joblib.load(model_path)
        logger.info("Modelo carregado de %s", model_path)
    else:
        logger.warning(
            "Modelo não encontrado em %s — execute 'make train' primeiro", model_path
        )
    yield
    _model = None


app = FastAPI(
    title="Customer Churn API",
    description="API de inferência de churn para a operadora de telecomunicações.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "path=%s method=%s status=%s latency_ms=%s",
        request.url.path,
        request.method,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=ChurnResponse, tags=["inferência"])
def predict(payload: ChurnRequest) -> ChurnResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute 'make train' antes de iniciar a API.",
        )

    row = {_FIELD_TO_COL[k]: v for k, v in payload.model_dump().items()}
    df = pd.DataFrame([row])

    try:
        proba = float(_model.predict_proba(df)[0, 1])
    except Exception as exc:
        logger.exception("Erro na predição: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChurnResponse(churn_probability=proba, churn_label=proba >= 0.5)
