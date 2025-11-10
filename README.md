# Brain-to-text-Team60
This repository is created for Team60's brain-to-text compeition in Data Mining Class

Dataset
The dataset for this competition consists of 10,948 sentences spoken by a single research participant, as described in Card et al., “An Accurate and Rapidly Calibrating Speech Neuroprosthesis” (2024, New England Journal of Medicine) (https://www.nejm.org/doi/full/10.1056/NEJMoa2314132). For each sentence, the dataset provides the transcript of the participant’s intended speech along with the corresponding time series of neural spiking activity recorded from 256 microelectrodes in the speech motor cortex. The data are organized into predefined train, validation, and test partitions. The train and validation sets include sentence labels, though you may repartition them if desired. The test set does not include labels and will be used for competition evaluation.

Evaluation
Decoding performance will be measured using aggregate Word Error Rate (WER) across the 1,450 held-out test sentences (data_test.hdf5). WER calculates the number of word substitutions, insertions, and deletions needed to match the decoded sentence to the true sentence, divided by the total number of words. Words are sequences of characters separated by spaces; all characters must match exactly (including apostrophes). Punctuation is ignored.

The test set is split for leaderboard purposes: 1/3 public and 2/3 private. The private leaderboard, which determines prizes, will be revealed after the competition ends (Dec 31, 2025).

Submissions should be a .csv file with id (0–1449) and text columns, ordered chronologically by session, block, and trial. Example scripts and baseline submissions are provided for formatting guidance.

Discussion of strategies and questions can be done on the Kaggle Discussion tab.

Link below is the research related to the competition
www.kaggle.com/competitions/brain-to-text-25/overview/citation

## Setup

Install the dependencies into your preferred virtual environment:

```bash
pip install -r requirements.txt
```

All scripts assume that raw competition assets (`data_train.hdf5`, `data_val.hdf5`, `data_test.hdf5`) live under a local `data/` directory at the repository root:

```
Brain-to-text-Team60/
├── data/
│   ├── data_train.hdf5
│   ├── data_val.hdf5
│   └── data_test.hdf5
```

## Preprocessing

`post_process_dataset/preprocess.py` converts the large HDF5 files into compressed `.npz` bundles that contain dense feature matrices plus the original transcripts.

Example commands (mean-pooling features):

```bash
python post_process_dataset/preprocess.py \
  --input data/data_train.hdf5 \
  --output data/processed/train_meanpool.npz \
  --method meanpool \
  --window 20 \
  --stride 20

python post_process_dataset/preprocess.py \
  --input data/data_val.hdf5 \
  --output data/processed/val_meanpool.npz \
  --method meanpool \
  --window 20 \
  --stride 20

python post_process_dataset/preprocess.py \
  --input data/data_test.hdf5 \
  --output data/processed/test_meanpool.npz \
  --method meanpool \
  --window 20 \
  --stride 20
```

You can switch to `--method spectral` for FFT-based power features or `--method flatten` to keep the raw timeseries.

## Training baseline decoders

Training is driven by YAML configuration files stored under `model_of_decoding/configs/`. Start with `baseline_knn.yaml` and adjust paths or hyperparameters as needed.

```bash
python model_of_decoding/train.py --config model_of_decoding/configs/baseline_knn.yaml
```

Artifacts are saved within the `output_dir` defined in the config:

- `models/<model_name>.joblib`: serialized decoder.
- `predictions/val_predictions.jsonl`: top-k validation candidates.
- `metrics/validation_metrics.json`: WER and exact match statistics.

## Generating a Kaggle submission

After training, point the inference script at the processed test bundle and the saved model:

```bash
python model_of_decoding/predict.py \
  --model-path outputs/baseline_knn/models/nearest_neighbor.joblib \
  --feature-bundle data/processed/test_meanpool.npz \
  --output submissions/nearest_neighbor_meanpool.csv \
  --top-k 3 \
  --candidates-json submissions/nearest_neighbor_candidates.jsonl
```

Upload the resulting CSV (`id,text` columns) to the Kaggle competition page.

## Evaluation utilities

`evaluation/wer.py` exposes reusable functions for computing Word Error Rate that match the public leaderboard metric. You can import `word_error_rate` inside notebooks or additional experiments.

## Reporting checklist

For the November 10 progress report, make sure the notebook or slide deck references:

- Preprocessing variants tried and their rationale.
- Models attempted (e.g., nearest-neighbor, logistic decoder) alongside runtime and validation WER.
- Screenshots of public leaderboard submissions generated via the scripts above.
- Planned next steps (hyperparameter sweeps, richer neural encoders, ensembling, etc.).