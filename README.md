# RRHO2 Python port

A Python port of the [R package RRHO2](https://github.com/RRHO2/RRHO2). Please cite the original authors of the R package if you use this Python port in published work. 
See the [Citation](#citation) section below.

Given two ranked gene lists, RRHO2 tests the overlap between every pair of
prefixes and reports the result as a map with four meaningful quadrants:
concordant up-up and down-down, and discordant up-down and down-up.

## Install

Install directly from GitHub with `uv`:

```bash
uv pip install "rrho2[plot] @ git+https://github.com/williamgilpin/rrho2"
```

or with `pip`:

```bash
pip install "rrho2[plot] @ git+https://github.com/williamgilpin/rrho2"
```

To add it as a dependency of a `uv` project:

```bash
uv add "rrho2[plot] @ git+https://github.com/williamgilpin/rrho2"
```

The `[plot]` extra pulls in matplotlib for `heatmap()` and `venn()`. Drop it if
you only need the overlap map as an array:

```bash
uv pip install "rrho2 @ git+https://github.com/williamgilpin/rrho2"
```

Requires Python 3.9+. Core dependencies are numpy and scipy.

To pin a specific commit, tag, or branch, append `@<ref>`:

```bash
uv pip install "rrho2[plot] @ git+https://github.com/williamgilpin/rrho2@v1.0.0"
```

### Development install

To hack on the package or run the test suite, clone it and install in editable
mode:

```bash
git clone https://github.com/williamgilpin/rrho2
cd rrho2
uv pip install -e ".[test]"
```

The `[test]` extra adds pytest, matplotlib, and pandas. With plain `pip`, use
`pip install -e ".[test]"`.

## Input format

Two gene lists, each pairing an identifier with a score:

- The score is conventionally `-log10(pvalue) * sign(effectSize)`, so
  up-regulated genes score positive and down-regulated genes negative.
- No missing values, unless you pass `drop_nan=True` (see below).
- Identifiers must be unique within each list. The two lists need not hold the
  same genes — they are reduced to the ones they share.
- Each list should contain both positive and negative scores. If one does not,
  the map is still built but two of its four quadrants cannot exist (see
  [Single-signed scores](#single-signed-scores)).

A list may be a pandas `DataFrame` (identifiers in the first column, scores in
the second), an `(n, 2)` array, or a `(names, values)` pair.

### Missing values

By default a `nan` score is an error, since a missing value has no rank. To drop
those genes instead:

```python
result = rrho2(list1, list2, drop_nan=True)
print(result.n_dropped, "genes dropped;", result.n_genes, "ranked")
```

Because RRHO2 ranks the *same* gene set twice, a gene whose score is missing in
**either** list is dropped from **both**, and the map is built on the surviving
intersection. This is exactly equivalent to never having passed those genes: the
hypergeometric population becomes the reduced size, so p-values stay correctly
calibrated rather than being computed against a universe that includes genes the
map cannot rank.

Note that this makes `drop_nan=True` slightly conservative — a gene measured
cleanly in list 1 is still discarded if list 2 is missing it. That is the
statistically honest choice for a rank-rank method, but if one list is much
patchier than the other, check `result.n_dropped` before reading much into the
map.

### Lists with different genes

The two lists do not have to contain the same genes. Genes present in only one
list are dropped, and the map is built on the shared set:

```python
result = rrho2(list1, list2)
print(result.n_unshared, "genes were in only one list;", result.n_genes, "compared")
```

As with `drop_nan`, this is exactly equivalent to intersecting the lists before
calling `rrho2`: the hypergeometric population is the size of the shared set, so
p-values are calibrated against the genes actually being ranked.

Two lists with *no* genes in common raise, since there is nothing to compare —
usually a sign the lists use different identifier types (symbols vs Ensembl IDs).
A large `n_unshared` is worth checking for the same reason.

### Single-signed scores

RRHO2 splits each list at zero to separate up- from down-regulated genes. If a
list is entirely positive (or entirely negative) there is no split, so two of the
four quadrants have no genes to rank. Rather than fail, `rrho2` builds the
quadrants that do exist, warns, and marks the others as empty:

```python
result.genelist_dd.peak      # None -- this quadrant cannot exist
result.genelist_dd.sizes     # (0, 0, 0)
```

Their cells are `nan` in `hypermat`, and `venn()` labels them rather than drawing
three misleading zeros. The usual cause is passing unsigned scores — raw
`-log10(pvalue)` without the `* sign(effectSize)` — in which case the fix is to
sign them, not to read the partial map.

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

- `hypermat` — the overlap map. Rows index list 1, columns index list 2, both
  running from most up-regulated to most down-regulated. For `method="hyper"`
  the values are `-log(p)`, or `-log10(p)` when `log10=True`; for
  `method="fisher"` they are log odds ratios. The white separator strips
  between quadrants are `nan`.
- `genelist_uu`, `genelist_dd`, `genelist_ud`, `genelist_du` — the genes at the
  most significant pixel of each quadrant. The first letter is the direction in
  list 1, the second in list 2. Each has `.list1`, `.list2`, `.overlap`,
  `.sizes`, and `.peak` (the pixel it was read from, or `None` if the quadrant
  cannot exist).
- `genelist(quadrant)` — the same, by name.
- `stepsize`, `boundary1`, `boundary2`, `strip1`, `strip2` — the grid geometry.
- `n_genes` — genes actually ranked, after all filtering.
- `n_dropped` — genes discarded for having a missing score (`drop_nan`).
- `n_unshared` — genes discarded for appearing in only one list.

## Key parameters

| parameter | default | meaning |
| --- | --- | --- |
| `stepsize` | `ceil(sqrt(n))` | genes between successive overlap tests |
| `log10` | `False` | report `-log10(p)` instead of `-log(p)` |
| `method` | `"hyper"` | `"hyper"` for p-values, `"fisher"` for log odds |
| `multiple_testing` | `"none"` | `"none"`, `"BH"`, or `"BY"` |
| `boundary` | `0.1` | width of the separator strip, as a fraction |
| `labels` | `None` | two names used to annotate plots |
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
ten scenarios. That validation is reproducible, but it needs the upstream R
sources, which are **not** redistributed here — this package is standalone, and
the R code belongs to the upstream GPL-3.0 project.

To run it yourself you need `R` on your `PATH` (no R packages required):

```bash
git clone --depth 1 https://github.com/RRHO2/RRHO2 /tmp/RRHO2
cp -r /tmp/RRHO2/R .
Rscript tests/r_reference/generate_reference.R
python -m pytest tests/ --run-r-comparison
```

`R/` and the generated ground truth are both gitignored. Alternatively, point
`RRHO2_R_SOURCE` at any directory containing `RRHO2_initialize.R` instead of
copying it into the repo:

```bash
RRHO2_R_SOURCE=/tmp/RRHO2/R Rscript tests/r_reference/generate_reference.R
```

Flags:

| flag | effect |
| --- | --- |
| *(none)* | run the comparison if ground truth exists, else skip with a notice |
| `--run-r-comparison` | require it; exit with a usage error if it is missing |
| `--no-r-comparison` | skip it even if the ground truth is present |

Use `--run-r-comparison` in CI so a missing oracle fails the build instead of
passing silently.

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
