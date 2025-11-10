# Team 60 – Brain-to-Text Progress Report (2025-11-10)

## 0. Progress Report 1 Recap (≤30 s)
- Established local development environment and cloned official competition starter kit.
- Completed exploratory analysis on a 5% slice of `data_train.hdf5` to understand channel statistics and sentence length distribution.
- Submitted the provided Kaggle starter baseline to validate submission formatting (baseline WER ≈ 94%).

## 1. New Preprocessing Pipelines
- Implemented `post_process_dataset/preprocess.py` to convert large HDF5 files into compressed `.npz` bundles with configurable feature extraction:
  - `meanpool`: temporal mean pooling with customizable window/stride.
  - `spectral`: FFT-based power spectrum features with frequency cutoffs.
  - `flatten`: direct vectorization for feature learning baselines.
- Script supports logging progress and can be parallelized externally (e.g., GNU parallel) for faster batch conversion.
- Outputs include aligned sample indices and transcripts to simplify downstream evaluation.

## 2. Modeling & Submission Attempts

| Attempt | Features | Model | Rationale | Runtime* | Public WER* | Insights |
|---------|----------|-------|-----------|----------|-------------|----------|
| A1 | `meanpool (window=20,stride=20)` | Nearest-neighbor (`metric=cosine`, `k=5`) | Fast retrieval-style baseline to establish reference point | ~8 min training (fit+index) | TBD (run after data placement) | Expect quick turnaround; acts as sanity check for pipeline |
| A2 | `meanpool` + StandardScaler | Logistic (SGD, multinomial) | Tests linear separability of pooled features; adds probabilistic outputs | ~25 min | TBD | Will log per-class probabilities for error analysis |
| A3 | `spectral (max_freq=120)` | Nearest-neighbor | Examines whether frequency-domain summarization improves distance retrieval | ~12 min | TBD | Compare WER vs meanpool to justify FFT cost |
| A4 | `flatten` + PCA (512 comps) | Logistic (SGD) | Higher-capacity linear model with dimensionality reduction | ~40 min | TBD | Requires PCA caching; expect better coverage of temporal nuance |

\*Run-time and WER placeholders will be updated once full training data are processed on the lab workstation. Scripts are in place to log actual numbers automatically.

### Automation
- Central training entrypoint: `python model_of_decoding/train.py --config model_of_decoding/configs/baseline_knn.yaml`
- Validation predictions and metrics auto-saved under `outputs/<run_name>/`.
- Inference & submission: `python model_of_decoding/predict.py --model-path outputs/<run>/models/<model>.joblib --feature-bundle data/processed/test_meanpool.npz --output submissions/<run>.csv`

## 3. Public Leaderboard Evidence
- Placeholder: insert screenshot of Kaggle submission once `submissions/<run>.csv` is uploaded.
- Recommended naming: `Team60_meanpool_knn.csv` to match competition rules.

## 4. Reflections & Next Steps
- The new pipeline modularizes feature extraction and modeling, enabling quick sweeps over methods without notebook duplication.
- Nearest-neighbor baseline provides interpretable errors (inspect retrieved transcripts) useful for qualitative analysis.
- Next actions:
  - Run full preprocessing on TA server (48-core) to populate `data/processed/`.
  - Execute attempts A1–A4, capture metrics, and update table above.
  - Explore lightweight neural decoder (1D CNN + CTC) using PyTorch Lightning for improved WER.
  - Investigate data augmentation (temporal jittering, channel dropout) to improve generalization.

## 5. Logistics
- All new code pushed under `model_of_decoding/`, `post_process_dataset/`, and `evaluation/`.
- Dependencies listed in `requirements.txt`; install via `python -m venv .venv && pip install -r requirements.txt`.
- Submission package instructions mirrored in repository README for team-wide reference.

