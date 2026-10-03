"""Pure retrieval metrics over ranked (doc_id, chunk_index) keys.

No I/O, no network, no database. Inputs are plain lists so the math stays
testable with hand-computed fixtures (see tests/test_metrics.py).

Relevance is binary: a retrieved chunk is a hit iff its key is in the gold
set. NDCG therefore degrades to rank-sensitive recall — documented, honest
at this scale.
"""

import math

Key = tuple[str, int]


def _gold_set(gold: list[Key]) -> set[Key]:
    return set(gold)


def hit_at_k(retrieved: list[Key], gold: list[Key], k: int) -> int:
    """1 if any gold key appears in the top-k, else 0."""
    goldset = _gold_set(gold)
    return 1 if any(r in goldset for r in retrieved[:k]) else 0


def recall_at_k(retrieved: list[Key], gold: list[Key], k: int) -> float:
    """Fraction of gold keys found in the top-k. Empty gold → 0.0 (caller
    should route negatives to the score-distribution path instead)."""
    if not gold:
        return 0.0
    goldset = _gold_set(gold)
    return sum(1 for r in retrieved[:k] if r in goldset) / len(goldset)


def precision_at_k(retrieved: list[Key], gold: list[Key], k: int) -> float:
    """Fraction of the top-k that is gold."""
    if k <= 0:
        return 0.0
    goldset = _gold_set(gold)
    return sum(1 for r in retrieved[:k] if r in goldset) / k


def reciprocal_rank(retrieved: list[Key], gold: list[Key]) -> float:
    """1/rank of the first hit, or 0.0 when nothing retrieved is gold."""
    goldset = _gold_set(gold)
    for i, r in enumerate(retrieved, 1):
        if r in goldset:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[Key], gold: list[Key], k: int) -> float:
    """Binary-relevance NDCG. IDCG assumes the ideal ranking puts all gold
    first, so with |gold| <= k a perfect retrieval scores exactly 1.0."""
    if not gold or k <= 0:
        return 0.0
    goldset = _gold_set(gold)
    dcg = sum(
        1.0 / math.log2(i + 1) for i, r in enumerate(retrieved[:k], 1) if r in goldset
    )
    ideal_hits = min(len(goldset), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ci(values: list[float]) -> tuple[float, float]:
    """(mean, 95% CI half-width) via the normal approximation.

    At n≈45 the half-width on recall≈0.7 is ≈±0.13 — the report prints it
    next to every mean so nobody over-reads a ±0.03 wiggle.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar on paired hit@k outcomes. b = hit→miss flips (regressions),
    c = miss→hit flips (fixes). Returns (stat, p) with continuity correction;
    stat ~ chi²(1), so p < 0.05 at stat > 3.84.

    Both runs score the same items, so between-item variance cancels — this
    is what makes a paired test valid where a raw-delta gate is noise.
    """
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(stat / 2))  # chi²(1) survival function
    return stat, p
