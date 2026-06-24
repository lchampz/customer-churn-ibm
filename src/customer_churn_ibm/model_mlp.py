"""MLP em PyTorch com wrapper sklearn-compatível."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

from customer_churn_ibm.config import SEED

logger = logging.getLogger(__name__)

torch.manual_seed(SEED)
np.random.seed(SEED)


class ChurnMLP(nn.Module):
    """Rede MLP para classificação binária de churn."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn-compatível para o ChurnMLP PyTorch.

    Permite usar o MLP diretamente em sklearn Pipelines e comparar com
    outros modelos via cross_val_score / GridSearchCV.
    """

    classes_ = np.array([0, 1])

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.model_: ChurnMLP | None = None

    # ------------------------------------------------------------------
    def _to_tensors(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[torch.Tensor, ...]:
        X_np = np.array(X, dtype=np.float32)
        X_t = torch.from_numpy(X_np)
        if y is not None:
            y_t = torch.from_numpy(np.array(y, dtype=np.float32))
            return X_t, y_t
        return (X_t,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

        split = int(0.85 * len(X_np))
        X_tr, X_val = X_np[:split], X_np[split:]
        y_tr, y_val = y_np[:split], y_np[split:]

        (X_tr_t, y_tr_t) = self._to_tensors(X_tr, y_tr)
        (X_val_t, y_val_t) = self._to_tensors(X_val, y_val)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        input_dim = X_np.shape[1]
        self.model_ = ChurnMLP(input_dim, self.hidden_dims, self.dropout).float()

        # Peso positivo para lidar com desbalanceamento
        pos_weight = torch.tensor(
            [(y_np == 0).sum() / max((y_np == 1).sum(), 1)], dtype=torch.float32
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_val_loss = float("inf")
        best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(
                        "Early stopping no epoch %d — val_loss=%.4f", epoch + 1, best_val_loss
                    )
                    break

        self.model_.load_state_dict(best_state)
        logger.info("Treinamento concluído. Best val_loss=%.4f", best_val_loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Chame fit() antes de predict_proba()")
        (X_t,) = self._to_tensors(X)
        self.model_.eval()
        with torch.no_grad():
            proba = torch.sigmoid(self.model_(X_t)).numpy()
        return np.column_stack([1 - proba, proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
