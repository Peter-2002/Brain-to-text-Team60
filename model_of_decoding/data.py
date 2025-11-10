"""Data loading helpers for preprocessing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class FeatureBundle:
    features: NDArray[np.float32]
    labels: Optional[np.ndarray]
    indices: NDArray[np.int32]

    @property
    def size(self) -> int:
        return self.features.shape[0]


def load_feature_bundle(
    npz_path: Path,
    feature_key: str = "features",
    label_key: str = "transcript",
    index_key: str = "indices",
) -> FeatureBundle:
    """Load a compressed feature bundle saved via `post_process_dataset.preprocess`."""
    with np.load(npz_path, allow_pickle=True) as bundle:
        features = bundle[feature_key].astype(np.float32)
        labels = bundle.get(label_key, None)
        if labels is not None:
            labels = labels.astype(object)
        indices = bundle.get(index_key)
        if indices is None:
            indices = np.arange(features.shape[0], dtype=np.int32)
        else:
            indices = indices.astype(np.int32)
    return FeatureBundle(features, labels, indices)

