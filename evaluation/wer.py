"""Utility functions for computing Word Error Rate (WER)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """Compute Levenshtein distance components between two token sequences.

    Returns the tuple (substitutions, insertions, deletions).
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
            )

    i, j = m, n
    substitutions = insertions = deletions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
            and ref[i - 1] != hyp[j - 1]
        ):
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            # Fallback if multiple paths share the same cost
            i = max(i - 1, 0)
            j = max(j - 1, 0)
    return substitutions, insertions, deletions


def tokenize(text: str) -> List[str]:
    """Tokenize on spaces while stripping punctuation per competition rules."""
    # The Kaggle competition ignores punctuation; we normalize to lowercase here.
    import re

    normalized = re.sub(r"[^\w\s']", " ", text.lower())
    tokens = [tok for tok in normalized.split() if tok]
    return tokens


@dataclass
class WERBreakdown:
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0
        return (self.substitutions + self.insertions + self.deletions) / self.reference_words


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> WERBreakdown:
    """Compute aggregate WER for aligned reference and hypothesis sentences."""
    total_sub = total_ins = total_del = total_ref = 0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)
        subs, ins, dels = _levenshtein(ref_tokens, hyp_tokens)
        total_sub += subs
        total_ins += ins
        total_del += dels
        total_ref += max(len(ref_tokens), 1)
    return WERBreakdown(total_sub, total_ins, total_del, total_ref)

