"""Utilities for transforming the raw HDF5 brain-to-text dataset into feature matrices.

This script implements a configurable preprocessing pipeline that
downsamples the neural time series, computes summary statistics or spectral
features, and persists the result in an .npz bundle that can be consumed by
the modeling scripts in `model_of_decoding/`.

Example usage
-------------

```bash
python post_process_dataset/preprocess.py \
    --input data/data_train.hdf5 \
    --output data/processed/train_meanpool.npz \
    --method meanpool \
    --window 20 \
    --stride 20
```

The same command can be applied to validation and test sets. All scripts
expect the data directory to reside under the repository root by default.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Iterable, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


def _mean_pooling(
    signal: NDArray[np.float32], window: int, stride: int
) -> NDArray[np.float32]:
    """Mean-pool the time dimension of a neural activity array.

    Parameters
    ----------
    signal:
        2-D array shaped `(time, channels)` containing the neural signal.
    window:
        Size of the pooling window in timesteps.
    stride:
        Stride between consecutive windows.
    """
    if signal.ndim != 2:
        raise ValueError("Expected 2-D signal (time, channels).")

    t, _ = signal.shape
    segments = []
    for start in range(0, max(t - window + 1, 1), stride):
        end = min(start + window, t)
        if end - start <= 0:
            break
        segments.append(signal[start:end].mean(axis=0, keepdims=True))
    if not segments:
        segments.append(signal.mean(axis=0, keepdims=True))
    pooled = np.concatenate(segments, axis=0)
    return pooled.flatten().astype(np.float32)


def _spectral_features(signal: NDArray[np.float32], max_freq: float = 120.0) -> NDArray[np.float32]:
    """Compute simple spectral power features using the FFT.

    The signal is detrended, transformed along the temporal axis, and
    truncated to retain low-frequency components that typically drive motor
    cortex activity.
    """
    if signal.ndim != 2:
        raise ValueError("Expected 2-D signal (time, channels).")

    detrended = signal - signal.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(detrended, axis=0)
    freqs = np.fft.rfftfreq(detrended.shape[0])
    mask = freqs <= max_freq
    power = (np.abs(fft[mask]) ** 2).astype(np.float32)
    return power.flatten()


def _flatten(signal: NDArray[np.float32]) -> NDArray[np.float32]:
    """Flatten the raw time series into a single feature vector."""
    return signal.astype(np.float32).flatten()


def _yield_samples(
    h5_path: pathlib.Path,
) -> Iterable[Tuple[int, NDArray[np.float32], str]]:
    """Yield (index, neural_activity, transcript) tuples from an HDF5 dataset."""
    with h5py.File(h5_path, "r") as handle:
        if "neural_activity" not in handle or "transcript" not in handle:
            raise KeyError(
                "The dataset must expose 'neural_activity' and 'transcript' entries."
            )
        neural = handle["neural_activity"]
        transcripts = handle["transcript"]
        for idx in range(neural.shape[0]):
            signal = np.array(neural[idx], dtype=np.float32)
            if signal.ndim == 1:
                # Some baselines store data as (channels, time); transpose to (time, channels)
                signal = signal.reshape(1, -1)
            transcript = transcripts[idx]
            if isinstance(transcript, bytes):
                transcript = transcript.decode("utf-8")
            yield idx, signal, transcript


def preprocess_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    method: str,
    window: int,
    stride: int,
    max_frequency: float,
) -> None:
    """Run preprocessing and persist the output .npz bundle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_list = []
    transcripts = []
    indices = []
    for idx, signal, transcript in _yield_samples(input_path):
        if method == "meanpool":
            features = _mean_pooling(signal, window, stride)
        elif method == "flatten":
            features = _flatten(signal)
        elif method == "spectral":
            features = _spectral_features(signal, max_frequency)
        else:
            raise ValueError(f"Unknown method '{method}'.")
        indices.append(idx)
        transcripts.append(transcript)
        feature_list.append(features)

        if idx % 50 == 0:
            LOGGER.info("Processed %d samples", idx)

    stacked = np.stack(feature_list)
    np.savez_compressed(
        output_path,
        features=stacked,
        transcript=np.array(transcripts, dtype=object),
        indices=np.array(indices, dtype=np.int32),
        method=method,
        window=window,
        stride=stride,
        max_frequency=max_frequency,
    )
    LOGGER.info("Saved features to %s (shape=%s)", output_path, stacked.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess brain-to-text HDF5 datasets.")
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Destination .npz path.")
    parser.add_argument(
        "--method",
        choices=("meanpool", "flatten", "spectral"),
        default="meanpool",
        help="Feature extraction strategy.",
    )
    parser.add_argument("--window", type=int, default=20, help="Window size for mean pooling.")
    parser.add_argument("--stride", type=int, default=20, help="Stride for mean pooling.")
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=120.0,
        help="Maximum FFT frequency retained when using spectral features.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    LOGGER.info("Starting preprocessing for %s", args.input)
    preprocess_file(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        window=args.window,
        stride=args.stride,
        max_frequency=args.max_frequency,
    )


if __name__ == "__main__":
    main()

