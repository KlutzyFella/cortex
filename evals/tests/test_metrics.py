"""Unit tests for metrics.py — every number hand-computed below.

Corpus: retrieved = [A, B, C, D] with gold = {B, D} (0-indexed ranks 2, 4).
"""

import math

import evalkit.metrics as metrics
from evalkit.metrics import (
    hit_at_k,
    mcnemar,
    mean_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

A = ("doc-1", 0)
B = ("doc-1", 1)
C = ("doc-2", 0)
D = ("doc-2", 1)
E = ("doc-3", 0)

RETRIEVED = [A, B, C, D]
GOLD = [B, D]


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, GOLD, 1) == 0  # top-1 is A, not gold
    assert hit_at_k(RETRIEVED, GOLD, 2) == 1  # B at rank 2
    assert hit_at_k(RETRIEVED, GOLD, 4) == 1


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert recall_at_k(RETRIEVED, GOLD, 2) == 0.5  # B only
    assert recall_at_k(RETRIEVED, GOLD, 4) == 1.0  # B and D
    assert recall_at_k(RETRIEVED, GOLD, 20) == 1.0  # k beyond list length


def test_recall_empty_gold_is_zero_not_nan():
    assert recall_at_k(RETRIEVED, [], 5) == 0.0


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert precision_at_k(RETRIEVED, GOLD, 2) == 0.5  # 1 of 2
    assert precision_at_k(RETRIEVED, GOLD, 4) == 0.5  # 2 of 4
    assert precision_at_k(RETRIEVED, GOLD, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, GOLD) == 0.5  # first hit B at rank 2
    assert reciprocal_rank([A, C, E], GOLD) == 0.0
    assert reciprocal_rank([B, A], GOLD) == 1.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([B, D, A, C], GOLD, 4) == 1.0


def test_ndcg_partial_ranking():
    # Retrieved [A, B, C, D] with gold {B, D}: hits at ranks 2 and 4.
    # DCG = 1/log2(3) + 1/log2(5); IDCG (2 gold) = 1/log2(2) + 1/log2(3).
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RETRIEVED, GOLD, 4) == expected


def test_ndcg_single_hit_discounted_by_rank():
    # Only B is gold, at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(RETRIEVED, [B], 4) == 1 / math.log2(3)


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k([A, C, E], GOLD, 4) == 0.0
    assert ndcg_at_k(RETRIEVED, [], 4) == 0.0
    assert ndcg_at_k(RETRIEVED, GOLD, 0) == 0.0


def test_mean_ci_marks_small_n_noise():
    mean, half = mean_ci([1.0] * 7 + [0.0] * 3)  # recall 0.7 at n=10
    assert mean == 0.7
    assert half > 0.25  # ±0.28: a 0.03 delta gate would live inside this


def test_mean_ci_empty_and_singleton():
    assert mean_ci([]) == (0.0, 0.0)
    assert mean_ci([0.5]) == (0.5, 0.0)


def test_mcnemar_no_discordants_is_p_one():
    assert mcnemar(0, 0) == (0.0, 1.0)


def test_mcnemar_consistent_regression_rejects():
    # 8 hit→miss, 0 miss→hit: stat = 7²/8 = 6.125 > 3.84.
    stat, p = mcnemar(8, 0)
    assert stat == 49 / 8
    assert p < 0.05


def test_mcnemar_split_flips_do_not_reject():
    # Same net count shift (6) as 8-vs-2… but inconsistent direction must not
    # gate-fail: stat = 5²/10 = 2.5 < 3.84.
    stat, p = mcnemar(8, 2)
    assert stat == 2.5
    assert p > 0.05


def test_chi2_survival_spot_values():
    # stat=3.84 is the p=0.05 boundary by construction.
    _, p = mcnemar(6, 1)  # stat = 16/7 ≈ 2.29
    assert p > 0.05
    stat, _ = mcnemar(6, 1)
    assert stat == 16 / 7
    assert metrics.mcnemar(10, 0)[1] < 0.01
