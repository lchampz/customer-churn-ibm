"""Script principal de treinamento: baselines + MLP PyTorch + registro no MLflow."""

import logging
import pathlib

import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from customer_churn_ibm.config import MLFLOW_EXPERIMENT, MODEL_DIR, SEED
from customer_churn_ibm.data import build_preprocessor, clean_data, get_splits, load_raw_data
from customer_churn_ibm.model_baseline import train_baselines
from customer_churn_ibm.model_mlp import MLPClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

np.random.seed(SEED)


def _evaluate_mlp(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def main() -> None:
    model_dir = pathlib.Path(MODEL_DIR)
    model_dir.mkdir(exist_ok=True)

    logger.info("=== Carregando dados ===")
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = get_splits(df)
    logger.info("Train=%d  Test=%d", len(X_train), len(X_test))

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    logger.info("=== Treinando baselines ===")
    results = train_baselines(build_preprocessor(), X_train, X_test, y_train, y_test)
    logger.info("\n%s", results.to_string(index=False))

    logger.info("=== Treinando MLP PyTorch ===")
    mlp_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", MLPClassifier(
            hidden_dims=(128, 64, 32),
            dropout=0.3,
            lr=1e-3,
            epochs=100,
            patience=15,
            batch_size=32,
        )),
    ])

    with mlflow.start_run(run_name="MLP_PyTorch"):
        mlp_pipeline.fit(X_train, y_train)
        metrics = _evaluate_mlp(mlp_pipeline, X_test, y_test)

        mlp_clf = mlp_pipeline.named_steps["model"]
        mlflow.log_params({
            "model": "MLP_PyTorch",
            "seed": SEED,
            "hidden_dims": str(mlp_clf.hidden_dims),
            "dropout": mlp_clf.dropout,
            "lr": mlp_clf.lr,
            "epochs": mlp_clf.epochs,
            "patience": mlp_clf.patience,
            "batch_size": mlp_clf.batch_size,
        })
        mlflow.log_metrics(metrics)

        logger.info(
            "MLP  F1=%.4f  AUC-ROC=%.4f  PR-AUC=%.4f",
            metrics["f1"],
            metrics["roc_auc"],
            metrics["pr_auc"],
        )

    artifact_path = model_dir / "mlp_pipeline.pkl"
    joblib.dump(mlp_pipeline, artifact_path)
    logger.info("Pipeline salvo em %s", artifact_path)


if __name__ == "__main__":
    main()
