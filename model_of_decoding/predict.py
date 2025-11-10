"""CLI tool for running inference and generating Kaggle submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_of_decoding.data import load_feature_bundle
from model_of_decoding.pipeline import export_submission, run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained decoder.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the serialized model produced by train.py.",
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        required=True,
        help="Path to the .npz feature bundle for inference (e.g. processed test set).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV file for Kaggle submission.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of candidates to keep per example.",
    )
    parser.add_argument(
        "--candidates-json",
        type=Path,
        help="Optional path to dump the raw candidate list (JSON lines).",
    )
    parser.add_argument(
        "--feature-key",
        default="features",
        help="Key used to access features within the .npz bundle.",
    )
    parser.add_argument(
        "--index-key",
        default="indices",
        help="Key used to access indices within the .npz bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_feature_bundle(
        args.feature_bundle,
        feature_key=args.feature_key,
        label_key="transcript",  # optional; inference bundles may not include labels
        index_key=args.index_key,
    )
    predictions = run_inference(args.model_path, bundle, top_k=args.top_k)
    primary = [candidates[0] if len(candidates) > 0 else "" for candidates in predictions]
    export_submission(primary, bundle.indices, args.output)

    if args.candidates_json:
        args.candidates_json.parent.mkdir(parents=True, exist_ok=True)
        with args.candidates_json.open("w", encoding="utf-8") as handle:
            for idx, candidates in zip(bundle.indices, predictions):
                handle.write(
                    json.dumps(
                        {
                            "index": int(idx),
                            "predictions": list(candidates),
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()

