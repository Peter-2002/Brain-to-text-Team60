"""Collection of baseline decoders for the brain-to-text task."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


class DecoderModel(ABC):
    """Abstract base class defining the interface for decoders."""

    @abstractmethod
    def fit(self, X: NDArray[np.float32], y: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: NDArray[np.float32], top_k: int = 1) -> List[Sequence[str]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "DecoderModel":
        raise NotImplementedError


@dataclass
class LogisticDecoder(DecoderModel):
    """Multinomial logistic regression trained with SGD."""

    max_iter: int = 2000
    alpha: float = 1e-4
    loss: str = "log_loss"
    penalty: str = "l2"
    encoder: LabelEncoder | None = None
    clf: SGDClassifier | None = None

    def fit(self, X: NDArray[np.float32], y: Sequence[str]) -> None:
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(list(y))
        self.clf = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=0,
            n_jobs=-1,
        )
        self.clf.fit(X, y_encoded)

    def predict(self, X: NDArray[np.float32], top_k: int = 1) -> List[Sequence[str]]:
        if self.clf is None or self.encoder is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        probs = self.clf.predict_proba(X)
        top_k = min(top_k, probs.shape[1])
        top_indices = np.argsort(probs, axis=1)[:, ::-1][:, :top_k]

        predictions: List[Sequence[str]] = []
        for row in top_indices:
            labels = self.encoder.inverse_transform(row)
            predictions.append(list(labels))
        return predictions

    def save(self, path: Path) -> None:
        if self.clf is None or self.encoder is None:
            raise RuntimeError("Nothing to save; fit the model first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "encoder": self.encoder}, path)

    @classmethod
    def load(cls, path: Path) -> "LogisticDecoder":
        state = joblib.load(path)
        instance = cls()
        instance.clf = state["clf"]
        instance.encoder = state["encoder"]
        return instance


@dataclass
class NearestNeighborDecoder(DecoderModel):
    """Feature retrieval baseline leveraging cosine similarity in embedding space."""

    metric: str = "cosine"
    n_neighbors: int = 5
    _nn: NearestNeighbors | None = None
    _train_features: NDArray[np.float32] | None = None
    _train_labels: np.ndarray | None = None

    def fit(self, X: NDArray[np.float32], y: Sequence[str]) -> None:
        self._train_features = np.asarray(X, dtype=np.float32)
        self._train_labels = np.asarray(list(y), dtype=object)
        self._nn = NearestNeighbors(metric=self.metric, n_neighbors=self.n_neighbors)
        self._nn.fit(self._train_features)

    def predict(self, X: NDArray[np.float32], top_k: int = 1) -> List[Sequence[str]]:
        if self._nn is None or self._train_labels is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        top_k = min(top_k, self.n_neighbors)
        distances, indices = self._nn.kneighbors(X, n_neighbors=top_k, return_distance=True)
        predictions: List[Sequence[str]] = []
        for dist_row, idx_row in zip(distances, indices):
            labels = self._train_labels[idx_row]
            predictions.append(list(labels))
        return predictions

    def save(self, path: Path) -> None:
        if self._train_features is None or self._train_labels is None or self._nn is None:
            raise RuntimeError("Nothing to save; fit the model first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "train_features": self._train_features,
                "train_labels": self._train_labels,
                "metric": self.metric,
                "n_neighbors": self.n_neighbors,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "NearestNeighborDecoder":
        state = joblib.load(path)
        instance = cls(metric=state["metric"], n_neighbors=state["n_neighbors"])
        instance._train_features = state["train_features"]
        instance._train_labels = state["train_labels"]
        instance._nn = NearestNeighbors(metric=instance.metric, n_neighbors=instance.n_neighbors)
        instance._nn.fit(instance._train_features)
        return instance


MODEL_REGISTRY: Dict[str, type[DecoderModel]] = {
    "logistic": LogisticDecoder,
    "nearest_neighbor": NearestNeighborDecoder,
}


def build_model(name: str, params: Dict[str, Any]) -> DecoderModel:
    """Instantiate a model from the registry."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**params)

