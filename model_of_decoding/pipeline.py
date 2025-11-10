"""Training and inference helpers wrapping the baseline models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from evaluation.wer import word_error_rate
from model_of_decoding import config as cfg
from model_of_decoding.data import FeatureBundle, load_feature_bundle
from model_of_decoding.models import DecoderModel, build_model


def _to_primary_candidate(preds: List[Sequence[str]]) -> List[str]:
    return [candidates[0] if len(candidates) > 0 else "" for candidates in preds]


def _compute_metrics(
    references: Sequence[str], hypotheses: Sequence[str]
) -> Dict[str, float]:
    breakdown = word_error_rate(references, hypotheses)
    total_ref = breakdown.reference_words
    total_err = breakdown.substitutions + breakdown.insertions + breakdown.deletions
    accuracy = np.mean([ref == hyp for ref, hyp in zip(references, hypotheses)])
    return {
        "wer": breakdown.wer,
        "wer_errors": float(total_err),
        "wer_ref_words": float(total_ref),
        "exact_match": float(accuracy),
    }


def _prepare_output_dir(root: Path) -> None:
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)


def train_and_evaluate(config: cfg.TrainingConfig) -> Dict[str, float]:
    """Train a decoder using the provided configuration."""
    _prepare_output_dir(config.output_dir)

    train_bundle = load_feature_bundle(
        config.data.train_npz,
        feature_key=config.data.feature_key,
        label_key=config.data.label_key,
        index_key=config.data.index_key,
    )
    if train_bundle.labels is None:
        raise ValueError("Training bundle must include transcripts.")

    model = build_model(config.model.name, config.model.params)
    model.fit(train_bundle.features, train_bundle.labels.tolist())

    metrics: Dict[str, float] = {}

    if config.data.val_npz:
        val_bundle = load_feature_bundle(
            config.data.val_npz,
            feature_key=config.data.feature_key,
            label_key=config.data.label_key,
            index_key=config.data.index_key,
        )
        if val_bundle.labels is None:
            raise ValueError("Validation bundle must include transcripts.")
        raw_predictions = model.predict(val_bundle.features, top_k=config.top_k)
        primary_predictions = _to_primary_candidate(raw_predictions)
        metrics = _compute_metrics(val_bundle.labels.tolist(), primary_predictions)

        pred_path = config.output_dir / "predictions" / "val_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as handle:
            for idx, reference, candidates in zip(
                val_bundle.indices, val_bundle.labels.tolist(), raw_predictions
            ):
                handle.write(
                    json.dumps(
                        {
                            "index": int(idx),
                            "reference": reference,
                            "predictions": list(candidates),
                        }
                    )
                    + "\n"
                )

        metrics_path = config.output_dir / "metrics" / "validation_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    model_path = config.output_dir / "models" / f"{config.model.name}.joblib"
    model.save(model_path)
    return metrics


def run_inference(
    model_path: Path,
    feature_bundle: FeatureBundle,
    top_k: int = 1,
) -> List[Sequence[str]]:
    """Load a serialized model and produce predictions for a feature bundle."""
    from model_of_decoding.models import MODEL_REGISTRY

    model_name = model_path.stem
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unable to infer model class from filename '{model_path.name}'. "
            f"Expected one of: {list(MODEL_REGISTRY)}. "
            "Please rename the file to <model_name>.joblib."
        )
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls.load(model_path)
    return model.predict(feature_bundle.features, top_k=top_k)


def export_submission(
    predictions: Sequence[str],
    indices: Sequence[int],
    output_path: Path,
) -> None:
    """Persist a Kaggle submission CSV with columns `id,text`."""
    import pandas as pd

    df = pd.DataFrame({"id": indices, "text": predictions})
    df = df.sort_values("id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

