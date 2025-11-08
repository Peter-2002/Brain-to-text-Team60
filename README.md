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
