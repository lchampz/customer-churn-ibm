"""Schemas Pydantic para request/response da API."""

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, description="Meses como cliente")
    monthly_charges: float = Field(..., ge=0, description="Cobrança mensal (USD)")
    total_charges: float = Field(..., ge=0, description="Total cobrado (USD)")
    gender: str = Field(..., description="Male / Female")
    senior_citizen: str = Field(..., description="Yes / No")
    partner: str = Field(..., description="Yes / No")
    dependents: str = Field(..., description="Yes / No")
    phone_service: str = Field(..., description="Yes / No")
    multiple_lines: str = Field(..., description="Yes / No / No phone service")
    internet_service: str = Field(..., description="DSL / Fiber optic / No")
    online_security: str = Field(..., description="Yes / No / No internet service")
    online_backup: str = Field(..., description="Yes / No / No internet service")
    device_protection: str = Field(..., description="Yes / No / No internet service")
    tech_support: str = Field(..., description="Yes / No / No internet service")
    streaming_tv: str = Field(..., description="Yes / No / No internet service")
    streaming_movies: str = Field(..., description="Yes / No / No internet service")
    contract: str = Field(..., description="Month-to-month / One year / Two year")
    paperless_billing: str = Field(..., description="Yes / No")
    payment_method: str = Field(
        ...,
        description=(
            "Electronic check / Mailed check / "
            "Bank transfer (automatic) / Credit card (automatic)"
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: bool
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
