"""Overlap statistics between two ranked lists, evaluated on a strided grid.

This is the Python counterpart of ``R/numericListOverlap.R`` and
``R/defaultStepSize.R``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import hypergeom

__all__ = [
    "default_step_size",
    "step_prefixes",
    "log_prefixes",
    "numeric_list_overlap",
]

#: Survival probabilities at or below this are recomputed in log space. Set
#: comfortably above the smallest normal float64 (~2.2e-308) so that denormals,
#: where the mantissa starts losing bits, never reach the fast path.
_MIN_NORMAL_SF = 1e-300


def default_step_size(n1: int, n2: int) -> int:
    """Resolution of the overlap grid: ``ceil(min(sqrt(n1), sqrt(n2)))``."""
    return int(np.ceil(min(np.sqrt(float(n1)), np.sqrt(float(n2)))))


def step_prefixes(n: int, stepsize: int) -> np.ndarray:
    """Prefix lengths at which overlaps are evaluated: ``1, 1+step, ... <= n``."""
    if stepsize < 1:
        raise ValueError("stepsize must be a positive integer")
    return np.arange(1, n + 1, stepsize, dtype=np.int64)


def log_prefixes(n: int, n_bins: Optional[int] = None) -> np.ndarray:
    """Prefix lengths spaced geometrically, densest at the top of the ranking.

    Linear spacing gives the top of the list -- where RRHO signal usually lives --
    a single grid point, while spending most of the grid on the uninformative
    tail. Geometric spacing inverts that: consecutive cutoffs near rank 1 differ
    by one gene, and near rank ``n`` by hundreds.

    Parameters
    ----------
    n
        Length of the ranked list.
    n_bins
        Requested number of grid points. Defaults to the count a linear grid
        would use, so the map costs the same to compute. The result may be
        shorter, because duplicate integer cutoffs are collapsed.

    Returns
    -------
    ndarray
        Strictly increasing prefix lengths, always starting at 1 and ending at
        the largest value ``<= n``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if n_bins is None:
        n_bins = len(step_prefixes(n, default_step_size(n, n)))
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if n == 1:
        return np.array([1], dtype=np.int64)

    # geomspace hits 1 and n exactly; rounding then collapses near-duplicates at
    # the dense end, so the grid is shorter than n_bins for small n.
    raw = np.geomspace(1.0, float(n), num=min(n_bins, n))
    return np.unique(np.rint(raw).astype(np.int64))


def _overlap_counts(
    names1: np.ndarray, names2: np.ndarray, prefix1: np.ndarray, prefix2: np.ndarray
) -> np.ndarray:
    """``counts[i, j] = |names1[:prefix1[i]] & names2[:prefix2[j]]|``.

    Evaluated with a two-dimensional cumulative sum rather than one set
    intersection per grid cell, so the cost is ``O(n log n + len1 * len2)``
    instead of ``O(n * len1 * len2)``. The counts are exact integers, so this
    is a pure speed change.
    """
    position_in_2 = {name: idx for idx, name in enumerate(names2)}
    # rank_in_2[i] is the 0-based position that names1[i] occupies in names2.
    rank_in_2 = np.fromiter(
        (position_in_2[name] for name in names1), dtype=np.int64, count=len(names1)
    )

    len1, len2 = len(prefix1), len(prefix2)
    # Bucket every element by the first prefix of each list that contains it.
    row_bucket = np.searchsorted(prefix1, np.arange(1, len(names1) + 1), side="left")
    col_bucket = np.searchsorted(prefix2, rank_in_2 + 1, side="left")

    # Elements past the last grid point belong to no prefix and never count.
    keep = (row_bucket < len1) & (col_bucket < len2)
    flat = row_bucket[keep] * len2 + col_bucket[keep]
    hist = np.bincount(flat, minlength=len1 * len2).reshape(len1, len2)
    return hist.cumsum(axis=0).cumsum(axis=1)


def _neglog_hypergeometric(
    counts: np.ndarray,
    prefix1: np.ndarray,
    prefix2: np.ndarray,
    n: int,
    population_offset: int = 1,
) -> np.ndarray:
    """``-log`` of the upper-tail hypergeometric p-value for each grid cell.

    The population size is ``n + population_offset``. ``population_offset=1``
    reproduces the R implementation, which calls
    ``phyper(count - 1, m = a, n = n - a + 1, k = b)`` and therefore models an
    urn holding one more element than the gene universe actually contains.
    ``population_offset=0`` is the statistically correct call.
    """
    draws_from_1 = prefix1[:, None]
    draws_from_2 = prefix2[None, :]
    population = n + population_offset
    quantile = counts - 1

    # Two paths, because scipy's hypergeometric routines have very different
    # characteristics (measured against exact rational arithmetic, see
    # docs/PORTING_NOTES.md):
    #
    #   sf    -- fully vectorised, ~13x faster, absolute error ~6e-14.
    #            Useless once the p-value underflows float64, which RRHO
    #            p-values routinely do (the 2000-gene example reaches
    #            -log(p) = 1383, and exp(-1383) is not representable).
    #   logsf -- loops in Python per cell, absolute error ~4e-12, but stays
    #            valid to arbitrarily small p.
    #
    # So take sf wherever it is comfortably inside the normal float64 range and
    # fall back to logsf only for the extreme cells. This is both faster and
    # more accurate than using logsf throughout.
    survival = hypergeom.sf(quantile, population, draws_from_1, draws_from_2)
    fast_path_ok = survival > _MIN_NORMAL_SF

    with np.errstate(divide="ignore"):
        result = -np.log(survival)

    extreme = ~fast_path_ok
    if extreme.any():
        rows, cols = np.nonzero(extreme)
        result[extreme] = -hypergeom.logsf(
            quantile[extreme], population, prefix1[rows], prefix2[cols]
        )
    return result


def _log_odds(
    counts: np.ndarray,
    prefix1: np.ndarray,
    prefix2: np.ndarray,
    n: int,
    offset: float = 1.0,
) -> np.ndarray:
    """Offset (Haldane-style) log odds ratio for each grid cell."""
    len_a = counts.astype(np.float64)
    len_b = prefix1[:, None].astype(np.float64)
    len_c = prefix2[None, :].astype(np.float64)
    numerator = (len_a + offset) * (n - len_b - len_c + len_a + offset)
    denominator = (len_c - len_a + offset) * (len_b - len_a + offset)
    # Both factors of the numerator are >= offset > 0 because
    # counts >= a + b - n, so the odds ratio is always positive and R's
    # log(abs(odds)) * sign(odds) collapses to log(odds).
    return np.log(numerator / denominator)


def numeric_list_overlap(
    names1,
    names2,
    stepsize: Optional[int] = None,
    method: str = "hyper",
    offset: float = 1.0,
    population_offset: int = 1,
    prefixes: Optional[np.ndarray] = None,
):
    """Overlap statistic over the strided grid of prefixes of two ranked lists.

    Parameters
    ----------
    names1, names2
        Element identifiers, each already sorted by its own ranking score. Both
        must contain the same identifiers.
    stepsize
        Number of elements between successive grid points. Ignored when
        ``prefixes`` is given; one of the two is required.
    prefixes
        Explicit, strictly increasing prefix lengths to evaluate, for a
        non-uniform grid (see :func:`log_prefixes`). Both lists use the same
        grid, as with ``stepsize``.
    method
        ``"hyper"`` for ``-log`` hypergeometric p-values, ``"fisher"`` for log
        odds ratios.
    offset
        Continuity offset used by ``method="fisher"``.
    population_offset
        See :func:`_neglog_hypergeometric`.

    Returns
    -------
    dict with keys ``counts`` and ``log_pval``, both ``(len1, len2)`` arrays
    indexed by grid point of list 1 and list 2 respectively.
    """
    names1 = np.asarray(names1)
    names2 = np.asarray(names2)
    n = len(names1)
    if len(names2) != n:
        raise ValueError("names1 and names2 must have the same length")

    if prefixes is not None:
        prefix1 = np.asarray(prefixes, dtype=np.int64)
        if prefix1.ndim != 1 or prefix1.size == 0:
            raise ValueError("prefixes must be a non-empty 1-D array")
        if np.any(np.diff(prefix1) <= 0):
            raise ValueError("prefixes must be strictly increasing")
        if prefix1[0] < 1 or prefix1[-1] > n:
            raise ValueError(f"prefixes must lie in [1, {n}]")
    elif stepsize is not None:
        prefix1 = step_prefixes(n, stepsize)
    else:
        raise ValueError("pass either stepsize or prefixes")
    prefix2 = prefix1
    counts = _overlap_counts(names1, names2, prefix1, prefix2)

    if method == "hyper":
        stat = _neglog_hypergeometric(counts, prefix1, prefix2, n, population_offset)
    elif method == "fisher":
        stat = _log_odds(counts, prefix1, prefix2, n, offset)
    else:
        raise ValueError(f"method must be 'hyper' or 'fisher', got {method!r}")

    return {"counts": counts, "log_pval": stat}
