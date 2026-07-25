"""Heatmap and Venn diagram for an :class:`~rrho2.core.RRHO2Result`.

Python counterpart of ``R/RRHO2_heatmap.R`` and ``R/RRHO2_vennDiagram.R``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .core import QUADRANTS, RRHO2Result

__all__ = ["jet_colormap", "heatmap", "venn_diagram"]

# The gradient R builds with colorRampPalette(); it interpolates in RGB, as
# LinearSegmentedColormap.from_list does.
_JET_ANCHORS = (
    "#00007F",
    "#0000FF",
    "#007FFF",
    "#00FFFF",
    "#7FFF7F",
    "#FFFF00",
    "#FF7F00",
    "#FF0000",
    "#7F0000",
)


def jet_colormap(n_colors: int = 101):
    """The rainbow gradient used by the R package, as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("rrho2_jet", _JET_ANCHORS, N=n_colors)
    # The nan separator strip renders white, as in R.
    try:
        return cmap.with_extremes(bad="white")
    except AttributeError:  # matplotlib < 3.6
        cmap.set_bad("white")
        return cmap


def heatmap(
    result: RRHO2Result,
    maximum: Optional[float] = None,
    minimum: Optional[float] = None,
    cmap=None,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    colorbar: bool = True,
    **imshow_kwargs,
):
    """Draw the RRHO2 map.

    Values are clipped to ``[minimum, maximum]`` when given, matching R. List 1
    runs along the x axis and list 2 along the y axis, both from most
    up-regulated (origin) to most down-regulated, as in R's ``image()``.

    Returns
    -------
    (matplotlib.axes.Axes, matplotlib.image.AxesImage)
    """
    import matplotlib.pyplot as plt

    hypermat = np.array(result.hypermat, dtype=np.float64, copy=True)
    if labels is None:
        labels = result.labels

    if maximum is not None:
        hypermat[hypermat > maximum] = maximum
    else:
        maximum = float(np.nanmax(hypermat))
    if minimum is not None:
        hypermat[hypermat < minimum] = minimum
    else:
        minimum = float(np.nanmin(hypermat))
    if minimum > maximum:
        raise ValueError("minimum > maximum, please check these function arguments!")

    if cmap is None:
        cmap = jet_colormap()

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))

    # R's image(z) puts the first index on x; imshow puts it on y, so transpose.
    image = ax.imshow(
        hypermat.T,
        origin="lower",
        cmap=cmap,
        vmin=minimum,
        vmax=maximum,
        aspect="auto",
        interpolation="nearest",
        **imshow_kwargs,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    if colorbar:
        if result.method == "hyper":
            title = "-log10(P-value)" if result.log10 else "-log(P-value)"
        else:
            title = "log Odds"
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.06, pad=0.03)
        bar.set_label(title)

    return ax, image


def venn_diagram(
    result: RRHO2Result,
    quadrant: str,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    colors: Sequence[str] = ("cornflowerblue", "darkorchid1"),
):
    """Two-set Venn diagram for the peak pixel of one quadrant.

    ``quadrant`` is one of ``"uu"``, ``"dd"``, ``"ud"``, ``"du"``: the first
    letter is the direction in list 1, the second in list 2. Circles are drawn
    at a fixed size, matching the R call's ``scaled = FALSE``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if quadrant not in QUADRANTS:
        raise ValueError(f"quadrant must be one of {QUADRANTS}, got {quadrant!r}")

    if labels is None:
        labels = result.labels if result.labels is not None else ("list1", "list2")
    genes = result.genelist(quadrant)
    n1, n2, n_overlap = genes.sizes

    direction = {"u": "Up", "d": "Down"}
    title = (
        f"{direction[quadrant[0]]} {labels[0]} "
        f"{direction[quadrant[1]]} {labels[1]}"
    )
    # matplotlib has no "darkorchid1"; R's value is #FF7FFF.
    palette = ["#FF7FFF" if c == "darkorchid1" else c for c in colors]

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.0))

    radius, shift = 1.0, 0.65
    for x, color in zip((-shift, shift), palette):
        ax.add_patch(Circle((x, 0.0), radius, facecolor=color, alpha=0.55, linewidth=0))

    ax.text(-shift - 0.45, 0, f"{n1 - n_overlap}", ha="center", va="center")
    ax.text(0, 0, f"{n_overlap}", ha="center", va="center")
    ax.text(shift + 0.45, 0, f"{n2 - n_overlap}", ha="center", va="center")
    ax.text(-shift, radius + 0.12, labels[0], ha="center", va="bottom", fontsize=11)
    ax.text(shift, radius + 0.12, labels[1], ha="center", va="bottom", fontsize=11)

    ax.set_title(title)
    ax.set_xlim(-shift - radius - 0.3, shift + radius + 0.3)
    ax.set_ylim(-radius - 0.2, radius + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
