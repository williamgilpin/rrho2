"""Benjamini-Hochberg / Benjamini-Yekutieli correction on -log p-values.

RRHO p-values routinely fall below the smallest positive float64 (~1e-308).
The R implementation round-trips through the linear scale
(``-log(p.adjust(exp(-neglog_p)))``), so every such cell underflows to ``p = 0``
and comes back as ``Inf`` -- which is exactly the situation its
"cannot handle extreme small p-values" warning describes.

:func:`adjust_neglog_pvalues` performs the same correction entirely in log
space, which agrees with R wherever R does not underflow and stays finite where
it does. :func:`legacy_adjust_neglog_pvalues` reproduces the R round-trip for
validation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["adjust_neglog_pvalues", "legacy_adjust_neglog_pvalues"]

_METHODS = ("BH", "BY")


def _log_correction_factor(method: str, m: int) -> float:
    """Extra ``log`` term that scales the ranked p-values."""
    if method == "BH":
        return 0.0
    if method == "BY":
        # sum(1 / 1..m), as in R's p.adjust(method = "BY")
        return float(np.log(np.sum(1.0 / np.arange(1, m + 1))))
    raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def adjust_neglog_pvalues(neglog_p: np.ndarray, method: str) -> np.ndarray:
    """Adjust ``-log(p)`` values, working entirely in log space.

    Mirrors R's ``p.adjust`` step-up procedure: sort p descending, scale the
    rank-``i`` value by ``m / i`` (times ``sum(1/1..m)`` for BY), take a
    running minimum, and clamp at 1. In ``-log`` space a running minimum of p
    is a running maximum, and clamping p at 1 is clamping ``-log(p)`` at 0.
    """
    factor = _log_correction_factor(method, neglog_p.size)
    flat = np.asarray(neglog_p, dtype=np.float64).ravel()
    m = flat.size

    # Ascending -log(p) == descending p, matching R's order(p, decreasing=TRUE).
    order = np.argsort(flat, kind="stable")
    ranks = np.arange(m, 0, -1, dtype=np.float64)

    adjusted = flat[order] + np.log(ranks) - np.log(float(m)) - factor
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.maximum(adjusted, 0.0)

    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    return out.reshape(np.shape(neglog_p))


def legacy_adjust_neglog_pvalues(neglog_p: np.ndarray, method: str) -> np.ndarray:
    """Reproduce the R round-trip through the linear scale, including ``Inf``."""
    factor = _log_correction_factor(method, np.size(neglog_p))
    flat = np.asarray(neglog_p, dtype=np.float64).ravel()
    m = flat.size

    pvalues = np.exp(-flat)
    order = np.argsort(-pvalues, kind="stable")
    ranks = np.arange(m, 0, -1, dtype=np.float64)

    adjusted = np.minimum.accumulate(np.exp(factor) * m / ranks * pvalues[order])
    adjusted = np.minimum(adjusted, 1.0)

    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    with np.errstate(divide="ignore"):
        return (-np.log(out)).reshape(np.shape(neglog_p))
