"""RRHO2: rank-rank hypergeometric overlap with four interpretable quadrants.

Python port of the R package RRHO2.

    >>> from rrho2 import rrho2
    >>> result = rrho2((genes, scores1), (genes, scores2), labels=("list1", "list2"))
    >>> result.hypermat.shape
    >>> result.genelist_dd.overlap

Reference
---------
Cahill, K. M., Huo, Z., Tseng, G. C., Logan, R. W., & Seney, M. L. (2018).
Improved identification of concordant and discordant gene expression signatures
using an updated rank-rank hypergeometric overlap approach.
*Scientific Reports*, 8(1), 1-11.
"""

from ._multitest import adjust_neglog_pvalues, legacy_adjust_neglog_pvalues
from ._overlap import (
    default_step_size,
    log_prefixes,
    numeric_list_overlap,
    step_prefixes,
)
from .core import (
    QUADRANTS,
    QuadrantGenes,
    RRHO2Result,
    rrho2,
    rrho2_initialize,
)

__version__ = "1.0.0"

__all__ = [
    "rrho2",
    "rrho2_initialize",
    "RRHO2Result",
    "QuadrantGenes",
    "QUADRANTS",
    "default_step_size",
    "numeric_list_overlap",
    "step_prefixes",
    "log_prefixes",
    "adjust_neglog_pvalues",
    "legacy_adjust_neglog_pvalues",
    "__version__",
]


def __getattr__(name):
    # Keep matplotlib an optional dependency: only import on first access.
    if name in ("heatmap", "venn_diagram", "jet_colormap"):
        from . import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
