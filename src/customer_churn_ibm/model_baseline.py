"""Modelos baseline: DummyClassifier e modelos sklearn clássicos."""

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from customer_churn_ibm.config import CV_FOLDS, MLFLOW_EXPERIMENT, SEED

logger = logging.getLogger(__name__)

BASELINE_CONFIGS: dict[str, dict] = {
    "DummyClassifier": {
        "model": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "log_artifact": False,
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=SEED, max_iter=1000, C=1.0),
        "log_artifact": True,
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1
        ),
        "log_artifact": True,
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=SEED),
        "log_artifact": True,
    },
}


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    has_proba = hasattr(pipeline.named_steps.get("model", pipeline[-1]), "predict_proba")
    y_proba = pipeline.predict_proba(X_test)[:, 1] if has_proba else None

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    } # pyright: ignore[reportAssignmentType]
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba) # type: ignore
        metrics["pr_auc"] = average_precision_score(y_test, y_proba)  # type: ignore

    return metrics


def train_baselines(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []

    for name, cfg in BASELINE_CONFIGS.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", cfg["model"])])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            metrics = _evaluate(pipeline, X_test, y_test)

            cv_f1 = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
            ).mean()

            mlflow.log_param("model", name)
            mlflow.log_param("seed", SEED)
            mlflow.log_metric("cv_f1_mean", cv_f1)
            mlflow.log_metrics(metrics)

            if cfg["log_artifact"]:
                mlflow.sklearn.log_model( # type: ignore
                    pipeline,
                    name.lower(),
                    input_example=X_test.head(1),
                )

            logger.info(
                "%-22s  F1=%.4f  AUC-ROC=%.4f  CV-F1=%.4f",
                name,
                metrics.get("f1", 0),
                metrics.get("roc_auc", 0),
                cv_f1,
            )
            rows.append({"model": name, "cv_f1": cv_f1, **metrics})

    return (
        pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    )
