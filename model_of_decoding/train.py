"""CLI entrypoint for training baseline decoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_of_decoding.config import load_config
from model_of_decoding.pipeline import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a brain-to-text decoder.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = train_and_evaluate(config)
    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("Training completed. No validation metrics because no validation set was provided.")


if __name__ == "__main__":
    main()

