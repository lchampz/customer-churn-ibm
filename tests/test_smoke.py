"""Smoke tests — verifica que módulos críticos carregam e instanciam corretamente."""

import numpy as np
import pandas as pd
import torch

from customer_churn_ibm.config import (CATEGORICAL_FEATURES,
                                       NUMERICAL_FEATURES, SEED)
from customer_churn_ibm.data import build_preprocessor
from customer_churn_ibm.model_mlp import ChurnMLP, MLPClassifier


def _synthetic_df(n: int = 20) -> pd.DataFrame:
    data = {col: [1.0] * n for col in NUMERICAL_FEATURES}
    for col in CATEGORICAL_FEATURES:
        data[col] = ["Yes"] * n
    return pd.DataFrame(data)


def test_seed_value():
    assert SEED == 42


def test_preprocessor_instantiates():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_preprocessor_fits_and_transforms():
    df = _synthetic_df()
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(df)
    assert result.shape[0] == len(df)
    assert result.shape[1] > 0


def test_churn_mlp_forward_pass():
    model = ChurnMLP(input_dim=10)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5,)


def test_mlp_classifier_fit_predict():
    clf = MLPClassifier(epochs=5, patience=3, batch_size=8)
    X = np.random.randn(60, 10).astype(np.float32)
    y = np.random.randint(0, 2, 60)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (60,)
    assert set(preds).issubset({0, 1})

    proba = clf.predict_proba(X)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
