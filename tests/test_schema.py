"""Testes de schema com pandera — valida contratos de dados."""

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from customer_churn_ibm.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "Tenure Months": pa.Column(float, pa.Check.ge(0), nullable=False),
        "Monthly Charges": pa.Column(float, pa.Check.ge(0), nullable=False),
        "Total Charges": pa.Column(float, pa.Check.ge(0), nullable=False),
        TARGET: pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=False,
)


def _make_df(n: int = 15) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Tenure Months": np.random.randint(1, 72, n).astype(float),
            "Monthly Charges": np.random.uniform(20.0, 120.0, n),
            "Total Charges": np.random.uniform(100.0, 8000.0, n),
            TARGET: np.random.randint(0, 2, n),
        }
    )
    for col in CATEGORICAL_FEATURES:
        df[col] = "Yes"
    return df


def test_valid_dataframe_passes_schema():
    RAW_SCHEMA.validate(_make_df())


def test_negative_tenure_fails():
    df = _make_df()
    df.loc[0, "Tenure Months"] = -5.0
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def test_invalid_target_value_fails():
    df = _make_df()
    df.loc[0, TARGET] = 2
    with pytest.raises(pa.errors.SchemaError):
        RAW_SCHEMA.validate(df)


def test_target_is_binary():
    df = _make_df(100)
    assert df[TARGET].isin([0, 1]).all()


def test_numerical_features_non_negative():
    df = _make_df(50)
    for col in NUMERICAL_FEATURES:
        assert (df[col] >= 0).all(), f"Coluna {col} tem valores negativos"
