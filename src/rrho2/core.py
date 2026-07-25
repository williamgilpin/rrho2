"""RRHO2: rank-rank hypergeometric overlap with four interpretable quadrants.

Python counterpart of ``R/RRHO2_initialize.R``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from ._multitest import adjust_neglog_pvalues, legacy_adjust_neglog_pvalues
from ._overlap import default_step_size, numeric_list_overlap, step_prefixes

__all__ = ["rrho2", "RRHO2Result", "QuadrantGenes", "QUADRANTS"]

#: The four quadrants, named by direction in list 1 then direction in list 2.
QUADRANTS = ("uu", "dd", "ud", "du")

_LOG10_E = float(np.log10(np.e))


@dataclass(frozen=True)
class QuadrantGenes:
    """Gene lists read off the most significant pixel of one quadrant."""

    list1: np.ndarray
    list2: np.ndarray
    overlap: np.ndarray
    #: 0-based position of the peak pixel within ``RRHO2Result.hypermat``.
    peak: Tuple[int, int]

    def __len__(self) -> int:
        return len(self.overlap)

    @property
    def sizes(self) -> Tuple[int, int, int]:
        return len(self.list1), len(self.list2), len(self.overlap)


@dataclass
class RRHO2Result:
    """Output of :func:`rrho2`.

    Attributes
    ----------
    hypermat
        Overlap statistic map. Rows index list 1, columns index list 2, both
        running from the most up-regulated element to the most down-regulated.
        The white separator strips are ``nan``. For ``method="hyper"`` the
        values are ``-log(p)`` (or ``-log10(p)`` when ``log10=True``); for
        ``method="fisher"`` they are log odds ratios.
    genelist_uu, genelist_dd, genelist_ud, genelist_du
        Genes at the peak pixel of each quadrant.
    """

    hypermat: np.ndarray
    method: str
    labels: Optional[Tuple[str, str]]
    log10: bool
    genelist_uu: QuadrantGenes
    genelist_dd: QuadrantGenes
    genelist_ud: QuadrantGenes
    genelist_du: QuadrantGenes
    stepsize: int
    boundary1: int
    boundary2: int
    strip1: int
    strip2: int
    counts: Optional[np.ndarray] = field(default=None, repr=False)

    def genelist(self, quadrant: str) -> QuadrantGenes:
        """Gene lists for one of ``"uu"``, ``"dd"``, ``"ud"``, ``"du"``."""
        if quadrant not in QUADRANTS:
            raise ValueError(f"quadrant must be one of {QUADRANTS}, got {quadrant!r}")
        return getattr(self, f"genelist_{quadrant}")

    def heatmap(self, **kwargs):
        """Draw the RRHO2 heatmap. See :func:`rrho2.plotting.heatmap`."""
        from .plotting import heatmap

        return heatmap(self, **kwargs)

    def venn(self, quadrant: str, **kwargs):
        """Draw a two-set Venn diagram. See :func:`rrho2.plotting.venn_diagram`."""
        from .plotting import venn_diagram

        return venn_diagram(self, quadrant, **kwargs)


def _is_sequence(obj) -> bool:
    return hasattr(obj, "__len__") and not isinstance(obj, (str, bytes))


def _as_gene_list(obj, argname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce a caller-supplied gene list into ``(names, values)``.

    Accepts a pandas ``DataFrame`` (first two columns), a ``(names, values)``
    pair, or an ``(n, 2)`` array-like.
    """
    if hasattr(obj, "iloc") and getattr(obj, "ndim", None) == 2:
        if obj.shape[1] < 2:
            raise ValueError(f"{argname} needs at least 2 columns")
        return np.asarray(obj.iloc[:, 0]), np.asarray(obj.iloc[:, 1], dtype=np.float64)

    if (
        isinstance(obj, (tuple, list))
        and len(obj) == 2
        and all(_is_sequence(part) for part in obj)
        and len(obj[0]) == len(obj[1])
    ):
        return np.asarray(obj[0]), np.asarray(obj[1], dtype=np.float64)

    arr = np.asarray(obj, dtype=object)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 0], arr[:, 1].astype(np.float64)

    raise TypeError(
        f"{argname} must be a DataFrame, an (n, 2) array, or a (names, values) pair"
    )


def _validate(names: np.ndarray, values: np.ndarray, argname: str) -> None:
    if len(names) == 0:
        raise ValueError(f"{argname} is empty")
    unique, counts = np.unique(names, return_counts=True)
    if np.any(counts > 1):
        dup = unique[counts > 1][:5]
        raise ValueError(f"Non-unique gene identifier found in {argname}: {list(dup)}")
    if np.any(np.isnan(values)):
        raise ValueError(f"NA value exists in {argname}, please remove them.")


def _peak_pixel(mat: np.ndarray, rows: slice, cols: slice) -> Tuple[int, int]:
    """0-based index of the largest cell in a submatrix, ties broken as in R.

    R locates the peak with ``which(max(quadrant) == hypermat, arr.ind = TRUE)``
    and takes the first row, so ties resolve in column-major order: lowest
    column first, then lowest row.
    """
    sub = mat[rows, cols]
    peak = np.nanmax(sub)
    # argwhere on the transpose scans columns before rows.
    col, row = np.argwhere(sub.T == peak)[0]
    return int(rows.start + row), int(cols.start + col)


def _ordered_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Elements of ``first`` also in ``second``, keeping ``first``'s order.

    Matches R's ``intersect`` for the unique identifiers RRHO2 requires.
    """
    other = set(second.tolist())
    return np.array([g for g in first.tolist() if g in other], dtype=first.dtype)


def rrho2(
    list1,
    list2,
    stepsize: Optional[int] = None,
    labels: Optional[Sequence[str]] = None,
    log10: bool = False,
    multiple_testing: str = "none",
    boundary: float = 0.1,
    method: str = "hyper",
    population_offset: int = 1,
    log_space_padjust: bool = True,
    return_counts: bool = False,
) -> RRHO2Result:
    """Build the RRHO2 overlap map for two ranked gene lists.

    Parameters
    ----------
    list1, list2
        Gene lists as a pandas ``DataFrame`` (identifier column, then score
        column), an ``(n, 2)`` array-like, or a ``(names, values)`` pair. Scores
        are typically ``-log10(pvalue) * sign(effect)``. Both lists must contain
        exactly the same identifiers, in any order, with no missing values.
    stepsize
        Elements between successive overlap tests. Defaults to
        ``ceil(sqrt(n))``.
    labels
        Two names used to annotate plots.
    log10
        Report ``-log10(p)`` instead of ``-log(p)``.
    multiple_testing
        ``"none"``, ``"BH"``, or ``"BY"``. Only valid for ``method="hyper"``.
    boundary
        Width of the white separator strip, as a fraction of the map size.
    method
        ``"hyper"`` for ``-log`` p-values, ``"fisher"`` for log odds ratios.
    population_offset
        Hypergeometric population size is ``n + population_offset``. The
        default of ``1`` reproduces the R package; ``0`` is the statistically
        correct choice.
    log_space_padjust
        Apply the multiple-testing correction in log space (default), which
        avoids the ``Inf`` cells R produces when a p-value underflows float64.
        Set ``False`` to reproduce R's behaviour exactly.
    return_counts
        Also return the raw overlap counts on the un-split grid.

    Returns
    -------
    RRHO2Result
    """
    if method not in ("hyper", "fisher"):
        raise ValueError(f"method must be 'hyper' or 'fisher', got {method!r}")
    if multiple_testing not in ("none", "BH", "BY"):
        raise ValueError(
            "multiple_testing must be one of 'none', 'BH', 'BY', got "
            f"{multiple_testing!r}"
        )
    if method == "fisher" and multiple_testing != "none":
        # R applies exp(-x) to the log odds ratio and feeds that to p.adjust,
        # which is meaningless. Refuse rather than reproduce it.
        raise ValueError(
            "multiple_testing is only meaningful for method='hyper'; log odds "
            "ratios are not p-values"
        )
    if not 0 <= boundary < 1:
        raise ValueError(f"boundary must be in [0, 1), got {boundary!r}")
    if labels is not None:
        labels = tuple(labels)
        if len(labels) != 2:
            raise ValueError("labels must have exactly 2 elements")

    names1, values1 = _as_gene_list(list1, "list1")
    names2, values2 = _as_gene_list(list2, "list2")
    _validate(names1, values1, "list1")
    _validate(names2, values2, "list2")
    if set(names1.tolist()) != set(names2.tolist()):
        raise ValueError("The gene names of the two lists must be identical.")

    # Descending score; a stable sort leaves ties in input order, as R's
    # order(decreasing = TRUE) does.
    order1 = np.argsort(-values1, kind="stable")
    order2 = np.argsort(-values2, kind="stable")
    names1, values1 = names1[order1], values1[order1]
    names2, values2 = names2[order2], values2[order2]

    n = len(names1)
    if stepsize is None:
        stepsize = default_step_size(n, len(names2))
    stepsize = int(stepsize)

    prefix1 = step_prefixes(n, stepsize)
    prefix2 = step_prefixes(len(names2), stepsize)
    len1, len2 = len(prefix1), len(prefix2)

    normal = numeric_list_overlap(
        names1, names2, stepsize, method=method, population_offset=population_offset
    )
    flipped = numeric_list_overlap(
        names1[::-1], names2, stepsize, method=method, population_offset=population_offset
    )
    hypermat_normal = normal["log_pval"]
    hypermat_flipx = flipped["log_pval"]

    if multiple_testing != "none":
        adjust = adjust_neglog_pvalues if log_space_padjust else legacy_adjust_neglog_pvalues
        hypermat_normal = adjust(hypermat_normal, multiple_testing)
        hypermat_flipx = adjust(hypermat_flipx, multiple_testing)

    strip1 = int(np.round(len1 * boundary))
    strip2 = int(np.round(len2 * boundary))

    # Grid points whose score is still positive, i.e. the up-regulated block.
    boundary1 = int(np.sum(values1[prefix1 - 1] > 0))
    boundary2 = int(np.sum(values2[prefix2 - 1] > 0))
    for value, total, which in ((boundary1, len1, "list1"), (boundary2, len2, "list2")):
        if value == 0:
            raise ValueError(
                f"No grid point of {which} has a positive score, so the map has no "
                "up-regulated quadrant. Check the sign convention of the scores."
            )
        if value == total:
            raise ValueError(
                f"Every grid point of {which} has a positive score, so the map has no "
                "down-regulated quadrant. Check the sign convention of the scores."
            )

    up1, up2 = slice(0, boundary1), slice(0, boundary2)
    down1 = slice(strip1 + boundary1, strip1 + len1)
    down2 = slice(strip2 + boundary2, strip2 + len2)

    hypermat = np.full((len1 + strip1, len2 + strip2), np.nan, dtype=np.float64)
    # quadrant III: up in 1, up in 2
    hypermat[up1, up2] = hypermat_normal[:boundary1, :boundary2]
    # quadrant I: down in 1, down in 2
    hypermat[down1, down2] = hypermat_normal[boundary1:, boundary2:]
    # quadrant II: up in 1, down in 2 -- read from the list-1-reversed map
    hypermat[up1, down2] = hypermat_flipx[len1 - boundary1 :, boundary2:][::-1]
    # quadrant IV: down in 1, up in 2
    hypermat[down1, up2] = hypermat_flipx[: len1 - boundary1, :boundary2][::-1]

    if np.any(np.isinf(hypermat[~np.isnan(hypermat)])):
        warnings.warn(
            "Inf was generated by the multiple testing procedure: some p-values are "
            "too small to represent after adjustment. Consider multiple_testing='none' "
            "or log_space_padjust=True.",
            RuntimeWarning,
            stacklevel=2,
        )

    if log10:
        hypermat = hypermat * _LOG10_E

    # 0-based offsets of each grid point into the sorted lists.
    start1 = prefix1 - 1
    start2 = prefix2 - 1

    def build(quadrant: str) -> QuadrantGenes:
        rows = up1 if quadrant[0] == "u" else down1
        cols = up2 if quadrant[1] == "u" else down2
        row, col = _peak_pixel(hypermat, rows, cols)
        if quadrant[0] == "u":
            genes1 = names1[: start1[row] + 1]
        else:
            genes1 = names1[start1[row - strip1] :]
        if quadrant[1] == "u":
            genes2 = names2[: start2[col] + 1]
        else:
            genes2 = names2[start2[col - strip2] :]
        return QuadrantGenes(
            list1=genes1,
            list2=genes2,
            overlap=_ordered_intersection(genes1, genes2),
            peak=(row, col),
        )

    return RRHO2Result(
        hypermat=hypermat,
        method=method,
        labels=labels,
        log10=log10,
        genelist_uu=build("uu"),
        genelist_dd=build("dd"),
        genelist_ud=build("ud"),
        genelist_du=build("du"),
        stepsize=stepsize,
        boundary1=boundary1,
        boundary2=boundary2,
        strip1=strip1,
        strip2=strip2,
        counts=normal["counts"] if return_counts else None,
    )


#: Alias matching the R function name.
rrho2_initialize = rrho2
