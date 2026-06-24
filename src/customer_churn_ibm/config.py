"""Constantes globais e configurações do projeto."""

SEED = 42

NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

TARGET = "Churn Value"

# Colunas removidas: identificadores, leakage e redundâncias geográficas
DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Reason",
    "Churn Score",  # leakage: score derivado do label
    "CLTV",         # leakage: calculado com base em churn esperado
]

TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_EXPERIMENT = "customer_churn_ibm"
MODEL_DIR = "models"
