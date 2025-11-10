"""Configuration dataclasses for decoding experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    train_npz: Path
    val_npz: Optional[Path] = None
    test_npz: Optional[Path] = None
    label_key: str = "transcript"
    feature_key: str = "features"
    index_key: str = "indices"


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]


@dataclass
class TrainingConfig:
    data: DataConfig
    model: ModelConfig
    output_dir: Path
    random_state: int = 42
    top_k: int = 1
    save_embeddings: bool = False


def _path_or_none(value: Optional[str]) -> Optional[Path]:
    if value in (None, "", "null"):
        return None
    return Path(value)


def load_config(path: Path) -> TrainingConfig:
    """Load a YAML configuration file into a TrainingConfig instance."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    data_section = payload["data"]
    model_section = payload["model"]
    output_dir = Path(payload["output_dir"])

    data_cfg = DataConfig(
        train_npz=Path(data_section["train_npz"]),
        val_npz=_path_or_none(data_section.get("val_npz")),
        test_npz=_path_or_none(data_section.get("test_npz")),
        label_key=data_section.get("label_key", "transcript"),
        feature_key=data_section.get("feature_key", "features"),
        index_key=data_section.get("index_key", "indices"),
    )
    model_cfg = ModelConfig(
        name=model_section["name"],
        params=model_section.get("params", {}),
    )

    return TrainingConfig(
        data=data_cfg,
        model=model_cfg,
        output_dir=output_dir,
        random_state=payload.get("random_state", 42),
        top_k=payload.get("top_k", 1),
        save_embeddings=payload.get("save_embeddings", False),
    )

