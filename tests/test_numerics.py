"""Numerical properties of the overlap engine.

These do not need R: they check internal consistency and the two-path
hypergeometric implementation against slower but independent references.
"""

import numpy as np
import pytest
from scipy.stats import hypergeom

from rrho2 import adjust_neglog_pvalues, numeric_list_overlap, step_prefixes
from rrho2._multitest import legacy_adjust_neglog_pvalues
from rrho2._overlap import _MIN_NORMAL_SF, _neglog_hypergeometric, _overlap_counts


def test_fast_and_slow_hypergeometric_paths_agree():
    """The vectorised sf path and the log-space fallback must not disagree.

    A visible seam between the two would show up as a discontinuity in the
    heatmap, so pin the maximum gap. Both are compared over the whole region
    where the fast path is valid.
    """
    n = 3000
    population = n + 1
    prefix = step_prefixes(n, 60)
    a = prefix[:, None]
    b = prefix[None, :]
    counts = np.minimum(a, b) * np.ones_like(a * b)
    # Sweep the overlap fraction so every regime is covered.
    for fraction in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0):
        counts = np.maximum(1, (np.minimum(a, b) * fraction).astype(np.int64))
        counts = np.minimum(counts, np.minimum(a, b))
        survival = hypergeom.sf(counts - 1, population, a, b)
        valid = survival > _MIN_NORMAL_SF
        if valid.sum() == 0:
            continue
        fast = -np.log(survival[valid])
        slow = -hypergeom.logsf(
            (counts - 1)[valid],
            population,
            np.broadcast_to(a, counts.shape)[valid],
            np.broadcast_to(b, counts.shape)[valid],
        )
        # ~4e-12 is scipy's logsf error floor; the fast path is the accurate one.
        np.testing.assert_allclose(fast, slow, rtol=0, atol=5e-11)


def test_extreme_cells_use_the_log_space_path():
    """Cells past float64 underflow must still produce large finite values."""
    n = 4000
    prefix = np.array([1000, 2000, 3000], dtype=np.int64)
    counts = np.array(
        [[1000, 1000, 1000], [1000, 2000, 2000], [1000, 2000, 3000]], dtype=np.int64
    )
    result = _neglog_hypergeometric(counts, prefix, prefix, n)
    assert np.isfinite(result).all()
    # Perfect overlap of 3000 of 4000 is astronomically unlikely.
    assert result[2, 2] > 745, "should exceed the float64 underflow point"
    assert result.max() < 1e6


def test_overlap_counts_match_brute_force():
    rng = np.random.default_rng(3)
    n = 500
    names1 = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    names2 = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    for stepsize in (1, 7, 23, 100, n):
        prefix1 = step_prefixes(n, stepsize)
        prefix2 = step_prefixes(n, stepsize)
        counts = _overlap_counts(names1, names2, prefix1, prefix2)
        assert counts.shape == (len(prefix1), len(prefix2))
        for i, p1 in enumerate(prefix1):
            for j, p2 in enumerate(prefix2):
                expected = len(set(names1[:p1]) & set(names2[:p2]))
                assert counts[i, j] == expected, (stepsize, i, j)


def test_counts_are_monotone_and_bounded():
    rng = np.random.default_rng(5)
    n = 400
    names1 = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    names2 = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    result = numeric_list_overlap(names1, names2, 13)
    counts = result["counts"]
    prefix = step_prefixes(n, 13)
    assert (np.diff(counts, axis=0) >= 0).all()
    assert (np.diff(counts, axis=1) >= 0).all()
    assert (counts <= np.minimum(prefix[:, None], prefix[None, :])).all()
    # Full prefixes of both lists overlap completely.
    assert counts[-1, -1] == len(set(names1[: prefix[-1]]) & set(names2[: prefix[-1]]))


def test_identical_lists_give_maximal_significance():
    n = 600
    names = np.array([f"g{i}" for i in range(n)], dtype=object)
    same = numeric_list_overlap(names, names, 25)["log_pval"]
    rng = np.random.default_rng(11)
    shuffled = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    random = numeric_list_overlap(names, shuffled, 25)["log_pval"]
    assert np.nanmax(same) > np.nanmax(random)
    assert (np.diag(same) >= np.diag(random) - 1e-9).all()


def test_neglog_pvalues_are_nonnegative():
    rng = np.random.default_rng(13)
    n = 800
    names1 = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    names2 = np.array([f"g{i}" for i in rng.permutation(n)], dtype=object)
    values = numeric_list_overlap(names1, names2, 29)["log_pval"]
    assert np.isfinite(values).all()
    assert (values >= -1e-12).all(), "a -log p-value cannot be meaningfully negative"


def test_numeric_list_overlap_rejects_bad_input():
    names = np.array(["a", "b", "c"], dtype=object)
    with pytest.raises(ValueError, match="same length"):
        numeric_list_overlap(names, names[:2], 1)
    with pytest.raises(ValueError, match="method"):
        numeric_list_overlap(names, names, 1, method="chisq")
    with pytest.raises(ValueError, match="stepsize"):
        numeric_list_overlap(names, names, 0)


# --------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["BH", "BY"])
def test_padjust_is_monotone_and_bounded(method):
    rng = np.random.default_rng(17)
    neglog_p = rng.uniform(0, 60, size=(20, 15))
    adjusted = adjust_neglog_pvalues(neglog_p, method)
    assert adjusted.shape == neglog_p.shape
    # Adjustment can only make p larger, i.e. -log(p) smaller.
    assert (adjusted <= neglog_p + 1e-9).all()
    assert (adjusted >= 0).all()
    # Order is preserved.
    flat_raw, flat_adj = neglog_p.ravel(), adjusted.ravel()
    order = np.argsort(flat_raw)
    assert (np.diff(flat_adj[order]) >= -1e-9).all()


@pytest.mark.parametrize("method", ["BH", "BY"])
def test_padjust_handles_extreme_values_without_inf(method):
    """The whole point of the log-space correction."""
    neglog_p = np.array([[5.0, 800.0], [1500.0, 0.5]])
    adjusted = adjust_neglog_pvalues(neglog_p, method)
    assert np.isfinite(adjusted).all()
    assert adjusted[1, 0] > adjusted[0, 1] > adjusted[0, 0]

    legacy = legacy_adjust_neglog_pvalues(neglog_p, method)
    assert np.isinf(legacy).any(), "legacy path should lose the extreme cells"


@pytest.mark.parametrize("method", ["BH", "BY"])
def test_padjust_ties_are_handled_consistently(method):
    neglog_p = np.array([3.0, 3.0, 3.0, 10.0, 1.0, 1.0])
    adjusted = adjust_neglog_pvalues(neglog_p, method)
    tied = adjusted[:3]
    assert np.allclose(tied, tied[0])


def test_bh_is_never_more_conservative_than_by():
    rng = np.random.default_rng(19)
    neglog_p = rng.uniform(0, 100, size=(12, 9))
    bh = adjust_neglog_pvalues(neglog_p, "BH")
    by = adjust_neglog_pvalues(neglog_p, "BY")
    assert (by <= bh + 1e-9).all()


def test_padjust_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        adjust_neglog_pvalues(np.array([1.0, 2.0]), "holm")
