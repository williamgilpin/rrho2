"""RRHO2: rank-rank hypergeometric overlap with four interpretable quadrants.

Python counterpart of ``R/RRHO2_initialize.R``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from ._multitest import adjust_neglog_pvalues, legacy_adjust_neglog_pvalues
from ._overlap import (
    default_step_size,
    log_prefixes,
    numeric_list_overlap,
    step_prefixes,
)

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
    #: 0-based position of the peak pixel within ``RRHO2Result.hypermat``, or
    #: ``None`` if the quadrant is empty because a list is single-signed.
    peak: Optional[Tuple[int, int]]

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
    #: Genes ranked in the map, after dropping missing and unshared genes.
    n_genes: int = 0
    #: Genes discarded because their score was missing in one or both lists.
    n_dropped: int = 0
    #: Genes discarded because they appeared in only one of the two lists.
    n_unshared: int = 0
    #: True when the rank cutoffs are geometrically rather than uniformly spaced.
    log_ranks: bool = False
    #: The rank cutoffs actually evaluated, for list 1 and list 2. Uniform unless
    #: ``log_ranks``; the single source of truth for :meth:`rank_cutoffs`.
    prefixes: Optional[Tuple[np.ndarray, np.ndarray]] = field(
        default=None, repr=False
    )
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

    # -- Array access, for plotting with something other than .heatmap() ----

    def quadrant_slices(self, quadrant: str) -> Tuple[slice, slice]:
        """``(rows, cols)`` slices locating one quadrant inside ``hypermat``.

        Rows index list 1, columns index list 2.
        """
        if quadrant not in QUADRANTS:
            raise ValueError(f"quadrant must be one of {QUADRANTS}, got {quadrant!r}")
        rows, cols = self.hypermat.shape
        up1 = slice(0, self.boundary1)
        up2 = slice(0, self.boundary2)
        down1 = slice(self.strip1 + self.boundary1, rows)
        down2 = slice(self.strip2 + self.boundary2, cols)
        return (
            up1 if quadrant[0] == "u" else down1,
            up2 if quadrant[1] == "u" else down2,
        )

    def quadrant_map(self, quadrant: str) -> np.ndarray:
        """One quadrant of ``hypermat`` as a standalone array, strips removed.

        A view, not a copy. ``[i, j]`` is the statistic for the top ``i``-th grid
        cutoff of list 1 against the top ``j``-th of list 2, where "top" follows
        the quadrant's direction: for ``"uu"`` both run from most up-regulated,
        for ``"dd"`` both run inward from most down-regulated.

        Empty (shape ``(0, n)``) when the quadrant cannot exist because a list is
        single-signed.
        """
        rows, cols = self.quadrant_slices(quadrant)
        return self.hypermat[rows, cols]

    def rank_cutoffs(self, quadrant: str) -> Tuple[np.ndarray, np.ndarray]:
        """Gene-count axis labels for :meth:`quadrant_map`.

        Returns ``(cutoffs1, cutoffs2)``: how many genes deep into each list the
        corresponding row/column of the quadrant reaches. Use these as tick
        labels -- element ``k`` labels row/column ``k`` of
        ``quadrant_map(quadrant)``.

        An up cutoff counts genes from the most up-regulated end, matching
        ``len(genelist(q).list1)``. A down cutoff is a *suffix* length counted
        from the most down-regulated end, so both axes measure distance from the
        quadrant's own origin.
        """
        rows, cols = self.quadrant_slices(quadrant)
        shape = self.hypermat.shape
        # Read the stored grid rather than re-deriving it from stepsize, which
        # would be wrong whenever the spacing is not uniform (log_ranks=True).
        if self.prefixes is not None:
            prefix1, prefix2 = self.prefixes
        else:
            prefix1 = prefix2 = np.arange(1, self.n_genes + 1, self.stepsize)

        def axis(
            sl: slice, limit: int, strip: int, is_up: bool, prefix: np.ndarray
        ) -> np.ndarray:
            indices = np.arange(*sl.indices(limit))
            if is_up:
                # Up block: row k is the grid point k, a prefix of the list.
                return prefix[indices]
            # Down block: the same offset core.py uses to build the gene lists,
            # turned into the length of the suffix it selects.
            return self.n_genes - prefix[indices - strip] + 1

        return (
            axis(rows, shape[0], self.strip1, quadrant[0] == "u", prefix1),
            axis(cols, shape[1], self.strip2, quadrant[1] == "u", prefix2),
        )

    def quadrant_peaks(self) -> dict:
        """``{quadrant: max statistic}`` for all four quadrants.

        ``nan`` for a quadrant that cannot exist. Useful for ranking concordant
        (``uu``, ``dd``) against discordant (``ud``, ``du``) signal.
        """
        peaks = {}
        for quadrant in QUADRANTS:
            block = self.quadrant_map(quadrant)
            peaks[quadrant] = (
                float(np.nanmax(block)) if block.size else float("nan")
            )
        return peaks


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


def _validate(
    names: np.ndarray, values: np.ndarray, argname: str, drop_nan: bool = False
) -> None:
    if len(names) == 0:
        raise ValueError(f"{argname} is empty")
    unique, counts = np.unique(names, return_counts=True)
    if np.any(counts > 1):
        dup = unique[counts > 1][:5]
        raise ValueError(f"Non-unique gene identifier found in {argname}: {list(dup)}")
    if not drop_nan and np.any(np.isnan(values)):
        n_nan = int(np.sum(np.isnan(values)))
        raise ValueError(
            f"NA value exists in {argname} ({n_nan} of {len(values)}). Remove them, "
            "or pass drop_nan=True to drop those genes from both lists."
        )


def _drop_nan_genes(
    names1: np.ndarray,
    values1: np.ndarray,
    names2: np.ndarray,
    values2: np.ndarray,
):
    """Drop genes whose score is ``nan`` in *either* list, from *both* lists.

    RRHO2 ranks the same gene set twice, so a gene can only be kept if it has a
    real score on both sides. Matching is by identifier, not position, because
    the two lists need not be in the same order.
    """
    bad = set(names1[np.isnan(values1)].tolist())
    bad |= set(names2[np.isnan(values2)].tolist())
    if not bad:
        return names1, values1, names2, values2, 0

    keep1 = np.array([name not in bad for name in names1.tolist()], dtype=bool)
    keep2 = np.array([name not in bad for name in names2.tolist()], dtype=bool)
    names1, values1 = names1[keep1], values1[keep1]
    names2, values2 = names2[keep2], values2[keep2]

    if len(names1) == 0:
        raise ValueError(
            "Every gene has a missing score in at least one list; nothing is left "
            "to compare."
        )
    return names1, values1, names2, values2, len(bad)


def _restrict_to_shared_genes(
    names1: np.ndarray,
    values1: np.ndarray,
    names2: np.ndarray,
    values2: np.ndarray,
):
    """Reduce both lists to the genes they have in common.

    RRHO2 compares two rankings of one gene set, so a gene present in only one
    list has no counterpart to be ranked against. Each list keeps its own order;
    only membership is filtered.
    """
    set1 = set(names1.tolist())
    set2 = set(names2.tolist())
    if set1 == set2:
        return names1, values1, names2, values2, 0

    shared = set1 & set2
    if not shared:
        raise ValueError(
            "The two lists have no gene identifiers in common, so there is nothing "
            "to compare. Check that both lists use the same identifier type "
            "(e.g. symbols vs Ensembl IDs)."
        )

    keep1 = np.array([name in shared for name in names1.tolist()], dtype=bool)
    keep2 = np.array([name in shared for name in names2.tolist()], dtype=bool)
    n_unshared = len(set1 ^ set2)
    return names1[keep1], values1[keep1], names2[keep2], values2[keep2], n_unshared


def _peak_pixel(
    mat: np.ndarray, rows: slice, cols: slice
) -> Optional[Tuple[int, int]]:
    """0-based index of the largest cell in a submatrix, ties broken as in R.

    R locates the peak with ``which(max(quadrant) == hypermat, arr.ind = TRUE)``
    and takes the first row, so ties resolve in column-major order: lowest
    column first, then lowest row.

    Returns ``None`` for a quadrant with no cells, which happens when a list is
    single-signed and so has no up- or no down-regulated block.
    """
    sub = mat[rows, cols]
    if sub.size == 0:
        return None
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
    drop_nan: bool = False,
    log_ranks: bool = False,
) -> RRHO2Result:
    """Build the RRHO2 overlap map for two ranked gene lists.

    Parameters
    ----------
    list1, list2
        Gene lists as a pandas ``DataFrame`` (identifier column, then score
        column), an ``(n, 2)`` array-like, or a ``(names, values)`` pair. Scores
        are typically ``-log10(pvalue) * sign(effect)``. Identifiers must be
        unique within each list, but the two lists need not hold the same genes:
        they are restricted to the genes they share, counted in
        ``result.n_unshared``. Missing scores are an error unless ``drop_nan``.
    stepsize
        Elements between successive overlap tests. Defaults to
        ``ceil(sqrt(n))``. Ignored when ``log_ranks=True``.
    log_ranks
        Space the rank cutoffs geometrically instead of uniformly, so the grid is
        dense at the top of each ranking and coarse in the tail. Useful when the
        overlap is concentrated in the first few hundred genes, which a linear
        grid compresses into one or two pixels. The number of grid points, and so
        the cost, is unchanged.

        The resulting ``hypermat`` is **not** evenly spaced in rank, so plot it
        against ``rank_cutoffs()`` rather than pixel index, and do not compare it
        cell-by-cell with a linear map. ``result.prefixes`` holds the cutoffs
        actually used.
    labels
        Two names used to annotate plots.
    log10
        Report ``-log10(p)`` instead of ``-log(p)``.
    multiple_testing
        Correct the p-values for the fact that the map runs one hypergeometric
        test per pixel -- a few thousand of them -- so the most extreme cell
        looks impressive by chance alone. Only valid for ``method="hyper"``.

        - ``"none"`` (default): raw, uncorrected p-values, as published RRHO2
          analyses use. Judge significance against the size of the grid; see the
          note on dependence below.
        - ``"BH"``: **Benjamini-Hochberg**, controls the *false discovery rate* --
          the expected share of flagged pixels that are false positives. The
          usual choice when you want a corrected map.
        - ``"BY"``: **Benjamini-Yekutieli**, also controls the false discovery
          rate but stays valid under arbitrary dependence between tests. It
          scales every p-value by an extra ``sum(1/1..m)`` (about 8.9 for a
          64x64 grid, growing slowly with grid size), so it is markedly more
          conservative than ``"BH"``.

        Both are step-up procedures: rank the ``m`` p-values, scale the ``k``-th
        smallest by ``m/k`` (times that extra factor for BY), then enforce
        monotonicity. Adjusted values are always larger than raw ones, so
        ``hypermat`` gets uniformly smaller.

        A caveat specific to RRHO: neighbouring pixels share almost all of their
        genes, so the tests are strongly dependent, not independent. BH assumes
        independence (or positive regression dependence) and BY is the
        dependence-robust alternative, but neither models this particular
        structure well -- BH is anti-conservative here and BY is conservative.
        Treat a corrected map as a guide, not an exact error rate. The
        correction is applied across the whole grid, before it is split into
        quadrants.
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
    drop_nan
        By default a missing score is an error. Set ``True`` to drop genes with a
        ``nan`` score instead. Because RRHO2 requires both lists to rank the same
        genes, a gene missing from *either* list is dropped from *both*, and the
        map is built on the surviving intersection. The hypergeometric population
        is the reduced size, so p-values stay correctly calibrated. The number
        dropped is reported as ``result.n_dropped``.

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
    _validate(names1, values1, "list1", drop_nan)
    _validate(names2, values2, "list2", drop_nan)

    n_dropped = 0
    if drop_nan:
        names1, values1, names2, values2, n_dropped = _drop_nan_genes(
            names1, values1, names2, values2
        )

    names1, values1, names2, values2, n_unshared = _restrict_to_shared_genes(
        names1, values1, names2, values2
    )

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

    if log_ranks:
        prefix1 = log_prefixes(n)
        prefix2 = log_prefixes(len(names2))
    else:
        prefix1 = step_prefixes(n, stepsize)
        prefix2 = step_prefixes(len(names2), stepsize)
    len1, len2 = len(prefix1), len(prefix2)

    grid = dict(method=method, population_offset=population_offset)
    if log_ranks:
        grid["prefixes"] = prefix1
    else:
        grid["stepsize"] = stepsize
    normal = numeric_list_overlap(names1, names2, **grid)
    flipped = numeric_list_overlap(names1[::-1], names2, **grid)
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
    # A single-signed list has no direction split, so two of the four quadrants
    # cannot exist. Build what does exist rather than refusing outright, but say
    # so: the usual cause is passing unsigned scores (raw -log10 p-values) and
    # forgetting to multiply by sign(effect).
    for value, total, which in ((boundary1, len1, "list1"), (boundary2, len2, "list2")):
        if value == 0:
            warnings.warn(
                f"No grid point of {which} has a positive score, so {which} has no "
                "up-regulated block and two of the four quadrants are empty. Scores "
                "should be signed, e.g. -log10(pvalue) * sign(effect).",
                RuntimeWarning,
                stacklevel=2,
            )
        elif value == total:
            warnings.warn(
                f"Every grid point of {which} has a positive score, so {which} has no "
                "down-regulated block and two of the four quadrants are empty. Scores "
                "should be signed, e.g. -log10(pvalue) * sign(effect).",
                RuntimeWarning,
                stacklevel=2,
            )

    up1, up2 = slice(0, boundary1), slice(0, boundary2)
    down1 = slice(strip1 + boundary1, strip1 + len1)
    down2 = slice(strip2 + boundary2, strip2 + len2)

    hypermat = np.full((len1 + strip1, len2 + strip2), np.nan, dtype=np.float64)
    # quadrant III: up in 1, up in 2
    hypermat[up1, up2] = hypermat_normal[:boundary1, :boundary2]
    # quadrant I: down in 1, down in 2
    hypermat[down1, down2] = hypermat_normal[boundary1:, boundary2:]
    # Quadrants II and IV read from the list-1-reversed map, so their row ranges
    # are mirrored. Written as explicit slices rather than open-ended ones,
    # because `flipx[len1 - 0:]` would be the whole array instead of nothing when
    # a list is single-signed and boundary1 collapses to 0 or len1.
    # quadrant II: up in 1, down in 2
    hypermat[up1, down2] = hypermat_flipx[len1 - boundary1 : len1, boundary2:][::-1]
    # quadrant IV: down in 1, up in 2
    hypermat[down1, up2] = hypermat_flipx[0 : len1 - boundary1, :boundary2][::-1]

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

    empty = np.array([], dtype=names1.dtype)

    def build(quadrant: str) -> QuadrantGenes:
        rows = up1 if quadrant[0] == "u" else down1
        cols = up2 if quadrant[1] == "u" else down2
        found = _peak_pixel(hypermat, rows, cols)
        if found is None:
            # The quadrant has no cells: one of the lists is single-signed.
            return QuadrantGenes(
                list1=empty, list2=empty, overlap=empty, peak=None
            )
        row, col = found
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
        n_genes=n,
        n_dropped=n_dropped,
        n_unshared=n_unshared,
        log_ranks=log_ranks,
        prefixes=(prefix1, prefix2),
        counts=normal["counts"] if return_counts else None,
    )


#: Alias matching the R function name.
rrho2_initialize = rrho2
