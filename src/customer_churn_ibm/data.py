"""Carregamento, limpeza e construção do pipeline de pré-processamento."""

import logging

import kagglehub
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn_ibm.config import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    SEED,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    logger.info("Baixando dataset via kagglehub")
    path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
    df = pd.read_excel(f"{path}/telco_customer_churn.xlsx")
    logger.info("Dataset carregado: %d linhas, %d colunas", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(0.0)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dados limpos: %d linhas, %d colunas", *df.shape)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def get_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
