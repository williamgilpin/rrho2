# RRHO2 Python port

A Python port of the [R package RRHO2](https://github.com/RRHO2/RRHO2). Please cite the original authors of the R package if you use this Python port in published work. 
See the [Citation](#citation) section below.

Given two ranked gene lists, RRHO2 tests the overlap between every pair of
prefixes and reports the result as a map with four meaningful quadrants:
concordant up-up and down-down, and discordant up-down and down-up.

## Install

```bash
uv pip install "rrho2[plot] @ git+https://github.com/williamgilpin/rrho2"
```

or with `pip`:

```bash
pip install "rrho2[plot] @ git+https://github.com/williamgilpin/rrho2"
```

The `[plot]` extra adds matplotlib, needed only for `heatmap()` and `venn()`.
Requires Python 3.9+; core dependencies are numpy and scipy.

For development, clone and install editable with the test extra:

```bash
uv pip install -e ".[test]"
```

## Input format

Two gene lists, each pairing a gene identifier with a score. The score is
conventionally `-log10(pvalue) * sign(effectSize)`, so up-regulated genes are
positive and down-regulated genes negative.

A list may be a pandas `DataFrame` (identifiers first, scores second), an
`(n, 2)` array, or a `(names, values)` pair. Identifiers must be unique within a
list; the two lists need not contain the same genes, and are reduced to those
they share. Gene order does not matter.

## Quick start

```python
import numpy as np
from rrho2 import rrho2

rng = np.random.default_rng(15213)
n_genes, n_de = 2000, 200
genes = np.array([f"Gene{i}" for i in range(n_genes)])

def scores():
    """200 up-regulated, 200 down-regulated, the rest noise."""
    up = -np.log10(rng.uniform(0, 0.05, n_de))
    down = np.log10(rng.uniform(0, 0.05, n_de))
    n_noise = n_genes - 2 * n_de
    noise = -np.log10(rng.uniform(0, 1, n_noise)) * rng.choice([1, -1], n_noise)
    return np.concatenate([up, down, noise])

result = rrho2(
    (genes, scores()),
    (genes, scores()),
    labels=("list1", "list2"),
    log10=True,
)

print(result.hypermat.shape)                    # the overlap map
print(result.genelist_dd.overlap[:10])          # genes down in both lists
print(len(result.genelist_uu.overlap))          # up in both

result.heatmap()        # needs matplotlib
result.venn("dd")
```

## Output

`rrho2` returns an `RRHO2Result`:

| attribute | what it holds |
| --- | --- |
| `hypermat` | the overlap map as a NumPy array; rows index list 1, columns list 2, both running most-up to most-down. Separator strips are `nan`. |
| `genelist_uu` / `_dd` / `_ud` / `_du` | genes at the most significant pixel of each quadrant (first letter = direction in list 1). Each has `.list1`, `.list2`, `.overlap`, `.sizes`, `.peak`. |
| `genelist(q)` | the same, by name |
| `quadrant_map(q)`, `rank_cutoffs(q)`, `quadrant_peaks()` | one quadrant as an array plus its axis labels, for plotting it yourself |
| `n_genes`, `n_dropped`, `n_unshared` | genes ranked, and how many were dropped as missing or unshared |
| `stepsize`, `boundary1`, `boundary2`, `strip1`, `strip2`, `prefixes` | grid geometry and the rank cutoffs used |

## Key parameters

| parameter | default | meaning |
| --- | --- | --- |
| `stepsize` | `ceil(sqrt(n))` | genes between successive overlap tests |
| `log10` | `False` | report `-log10(p)` instead of `-log(p)` |
| `method` | `"hyper"` | `"hyper"` for p-values, `"fisher"` for log odds |
| `multiple_testing` | `"none"` | `"none"`, `"BH"`, or `"BY"` |
| `boundary` | `0.1` | width of the separator strip, as a fraction |
| `labels` | `None` | two names used to annotate plots |
| `log_ranks` | `False` | space rank cutoffs geometrically, for resolution at the top of the ranking |
| `drop_nan` | `False` | drop genes with a missing score instead of erroring |
| `population_offset` | `1` | `1` matches R; `0` is statistically correct |
| `log_space_padjust` | `True` | avoid underflow in BH/BY; `False` matches R |

The last two exist because the R implementation has quirks worth being able to
opt out of. Both are documented in
[docs/PORTING_NOTES.md](docs/PORTING_NOTES.md).

## Tests

```bash
python -m pytest tests/
```

35 tests cover the API and the numerics and always run. A further 52 compare
against the original R implementation; those are optional and skip unless the
ground truth is present (see below). The run prints a clear notice when they
skip, so a green suite never quietly means "a third of it did not run".

## Validating against the R implementation

The port is validated cell-by-cell and gene-by-gene against real R output across
ten scenarios. Those tests need the upstream R sources, which are not
redistributed here, so they skip unless you fetch them (with `R` on your `PATH`;
no R packages required):

```bash
git clone --depth 1 https://github.com/RRHO2/RRHO2 /tmp/RRHO2
cp -r /tmp/RRHO2/R .
Rscript tests/r_reference/generate_reference.R
python -m pytest tests/ --run-r-comparison
```

`R/` and the generated ground truth are both gitignored. `--run-r-comparison`
makes a missing oracle a hard error instead of a skip, which is what you want in
CI; `--no-r-comparison` skips these tests even when the data is present.

## Relationship to the R package

This is a standalone Python package, not a wrapper: it has no R dependency at
runtime. On realistic transcriptome-sized inputs it runs 8-17x faster than the R
original. Two bugs in the R implementation are fixed, and both fixes can
be turned off to reproduce R exactly. Every difference, the measurements behind
it, and the reasoning are recorded in
[docs/PORTING_NOTES.md](docs/PORTING_NOTES.md).

## Credit

The RRHO2 method and its original R implementation are the work of Kelly M.
Cahill, Zhiguang Huo, and colleagues:
[github.com/RRHO2/RRHO2](https://github.com/RRHO2/RRHO2). This repository is an
independent Python port of that package; the algorithm is theirs, and if you use
it in published work, cite their papers, not this port.

## Citation

- Cahill, K. M., Huo, Z., Tseng, G. C., Logan, R. W., & Seney, M. L. (2018).
  Improved identification of concordant and discordant gene expression
  signatures using an updated rank-rank hypergeometric overlap approach.
  *Scientific Reports*, 8(1), 1-11.
- Plaisier, S. B., Taschereau, R., Wong, J. A., & Graeber, T. G. (2010).
  Rank-rank hypergeometric overlap: identification of statistically significant
  overlap between gene-expression signatures. *Nucleic Acids Research*, 38(17),
  e169.

## License

GPL-3.0. As a port of a GPL-3.0 package, this is a derivative work and carries
the same license.
