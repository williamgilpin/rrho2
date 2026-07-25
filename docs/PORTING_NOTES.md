# RRHO2: R to Python porting notes

This document records every way the Python package differs from the R original,
and what was measured to justify each change.

The upstream R sources are the validation oracle:
`tests/r_reference/generate_reference.R` runs them and writes ground truth that
`tests/test_against_r.py` compares against, cell by cell and gene by gene.

They are **not** redistributed with this package, which is standalone and has no
R dependency at runtime. See "Validating against the R implementation" in
README.md for how to fetch them and reproduce the comparison. Without them the
52 R-comparison tests skip; the remaining 35 always run.

## Scope

Ported: the core tool (`RRHO2_initialize` -> `rrho2`), the overlap engine
(`numericListOverlap`), `defaultStepSize`, and both plots.

The plots are reimplemented in matplotlib rather than being pixel-exact copies
of R's `image()` and `VennDiagram::draw.pairwise.venn()`. Layout and styling
differ; the data plotted does not.

## Name mapping

| R | Python |
| --- | --- |
| `RRHO2_initialize(list1, list2)` | `rrho2(list1, list2)` (alias `rrho2_initialize`) |
| `RRHO2_heatmap(obj)` | `result.heatmap()` / `rrho2.plotting.heatmap` |
| `RRHO2_vennDiagram(obj, type)` | `result.venn(quadrant)` / `rrho2.plotting.venn_diagram` |
| `numericListOverlap` | `rrho2.numeric_list_overlap` |
| `defaultStepSize` | `rrho2.default_step_size` |
| `log10.ind` | `log10` |
| `multipleTesting` | `multiple_testing` |
| `obj$hypermat` | `result.hypermat` |
| `obj$genelist_dd$gene_list_overlap_dd` | `result.genelist_dd.overlap` |

`result.genelist("dd")` also works, for programmatic access.

## Verified identical

Validated against R across ten scenarios (defaults, `log10`, custom
`stepsize`/`boundary`, Fisher, BH, BY, a high-signal case that drives R's
correction past float64 underflow, shuffled input order with an odd list
length, and heavily tied scores):

- `hypermat`, including the exact placement and size of the `nan` separator
  strips.
- All four quadrants' gene lists, in order, including the peak-pixel tie-break.
  R resolves ties via `which(..., arr.ind = TRUE)`, which scans column-major;
  `_peak_pixel` reproduces that by scanning the transpose.
- Descending sort with ties left in input order (R's stable
  `order(decreasing = TRUE)` -> `np.argsort(kind="stable")`).
- `defaultStepSize` over a grid of list lengths.
- `numericListOverlap` counts and statistics for both methods.
- Both multiple-testing corrections, including R's `Inf` cells, when
  `log_space_padjust=False`.

## Changes that do not affect results

### 1. Overlap counts via 2-D cumulative sum

R recomputes `sum(sample1[1:a] %in% sample2[1:b])` for every grid cell, which is
`O(n * len1 * len2)`. The port maps each gene to its rank in list 2, buckets the
genes by grid cell, and takes a 2-D cumulative sum: `O(n log n + len1 * len2)`.

Counts are exact integers, so this is purely a speed change. Verified against
brute-force set intersection for step sizes 1, 7, 23, 100, and `n`
(`test_overlap_counts_match_brute_force`).

### 2. Two-path hypergeometric evaluation

`scipy.stats.hypergeom` offers two relevant routines with quite different
characteristics. Measured against exact rational arithmetic (`math.comb` +
`mpmath` at 60 digits):

| routine | vectorised | absolute error | valid range |
| --- | --- | --- | --- |
| `sf` | yes | ~6e-14 | until p underflows float64 |
| `logsf` | no, loops per cell | ~4e-12 | any p |
| R's `phyper` | n/a | ~1e-15 relative | any p |

RRHO p-values routinely underflow: the 2000-gene example in the README reaches
`-log(p) = 1383`, and `exp(-1383)` is not representable. So `sf` alone is not
enough, but `logsf` alone is both slower and less accurate.

The port therefore uses `sf` wherever it exceeds `1e-300` (safely inside the
normal float64 range) and falls back to `logsf` only for the extreme cells.
`test_fast_and_slow_hypergeometric_paths_agree` pins the seam between the two
paths at 5e-11, well below anything visible in a heatmap.

### 3. Residual difference from R

The fallback path inherits scipy's ~4e-12 absolute error floor, where R's
`phyper` is accurate to ~1e-15 relative. Because the floor is constant in
absolute terms, it is negligible in relative terms exactly where it matters:

| `-log(p)` | max relative difference from R |
| --- | --- |
| > 1 | 5e-12 |
| > 10 | 1e-12 |
| > 100 | 1e-13 |

In other words the disagreement is confined to cells that are not remotely
significant, and every cell that drives an interpretation agrees with R to
about 13 significant digits (`test_significant_cells_match_r_to_near_machine_precision`).
Closing the last few digits would mean reimplementing `phyper`'s summation; it
would not change any output anyone reads.

### 4. Dead code dropped

`RRHO2_initialize` computes `N <- max(nlist1, nlist2)` and never uses it. It
also computes `hypermat_flipX2`, which is commented out. Neither is ported.

### 5. Fisher log odds simplified

R computes `log(abs(Odds)) * sign(Odds)`. With the default `offset = 1` both
factors of the numerator are at least 1 (because `count >= a + b - n`), so the
odds ratio is always positive and the expression collapses to `log(Odds)`. The
port computes `log` directly; the Fisher reference case confirms the values are
unchanged.

## Changes that do affect results

Each is verified by a test that asserts what moves and what does not.

### 6. Multiple testing in log space (default: on)

**What R does.** `p.adjust(exp(-hypermat), method = "BH")`, then `-log` of the
result. The round trip through the linear scale destroys any p-value below
`~1e-308`.

**Why it matters more than the `Inf` warning suggests.** R's own warning
mentions `Inf`, but precision starts degrading well before that, as soon as
`exp(-x)` becomes a denormal:

| `-log(p)` | `exp(-x)` | after round trip | error |
| --- | --- | --- | --- |
| 700 | 9.9e-305 | 700.0000 | 0 |
| 708 | 3.3e-308 | 708.0000 | 0 |
| 730 | 9.2e-318 | 730.0000 | 1.8e-07 |
| 740 | 4.2e-322 | 739.9974 | 2.6e-03 |
| 745 | 4.9e-324 | 744.4401 | 5.6e-01 |
| 746 | 0 | `Inf` | `Inf` |

**What the port does.** `adjust_neglog_pvalues` runs the identical step-up
procedure entirely in `-log` space: a running minimum of p is a running maximum
of `-log(p)`, and clamping p at 1 is clamping `-log(p)` at 0. Nothing underflows.

**Effect.** Identical to R (to 1e-10) for every cell whose p-value R can
represent; finite and correct where R returns `Inf` or a denormal-corrupted
value. On the `strong` reference case R loses cells outright to `Inf`.

Pass `log_space_padjust=False` to reproduce R exactly, including its `Inf`
cells and its warning. This is what the BH/BY reference comparisons use, so
both behaviours stay pinned.

### 7. `multiple_testing` with `method="fisher"` now raises

R feeds `exp(-log_odds_ratio)` to `p.adjust`. A log odds ratio is not a p-value
and can be negative, so the result is meaningless. The port raises `ValueError`
rather than reproduce it. `method="fisher"` with `multiple_testing="none"` is
unaffected and matches R.

### 8. Single-signed scores now raise (opt-out: none)

If no grid point has a positive score, R's `1:boundary1` becomes `1:0` — the
two-element vector `c(1, 0)` — and `(boundary1+1):len1` counts backwards.
The result is a silently malformed map rather than an error.

Both lists must contain positive and negative scores for the four quadrants to
exist, so the port raises `ValueError` with an explanation. This turns silent
corruption into a clear failure; it cannot change a previously-correct result.

### 9. `population_offset` (default: 1, matching R)

R calls `phyper(q = count - 1, m = a, n = n - a + 1, k = b)`. Since `m + n` is
the population size, this models an urn of `n + 1` genes when the universe holds
`n` — an off-by-one inherited from the original RRHO package.

The default `population_offset=1` reproduces R. `population_offset=0` is the
statistically correct call, and is verified against a direct `scipy.stats`
hypergeometric call. The correction shifts the peak `-log(p)` by well under 1%,
so it does not change interpretation, which is why it is opt-in rather than the
default: matching published RRHO2 output is the more useful default.

## Performance

Same machine, same synthetic data, `method="hyper"`, defaults otherwise.
`nDE` is 10% of genes in each direction unless noted.

| genes | R | Python | speedup |
| --- | --- | --- | --- |
| 15,000 (5% DE) | 5.12 s | 0.30 s | 17x |
| 20,000 (5% DE) | 9.67 s | 0.57 s | 17x |
| 20,000 (10% DE) | 9.79 s | 1.16 s | 8x |
| 50,000 (10% DE) | 69.75 s | 14.92 s | 4.7x |

The speedup narrows as the signal strengthens, because more cells fall past
float64 underflow and onto the per-cell `logsf` fallback (3% of cells at
15,000 genes, 46% at 50,000). Vectorising the log-space tail sum would recover
that, at the cost of hand-rolling a numerically delicate summation; it was not
judged worth the risk for the sizes real gene lists reach.

## Reproducing the validation

The R sources are gitignored, so fetch them first. Needs `R` on `PATH`; no R
packages are required.

```bash
git clone --depth 1 https://github.com/RRHO2/RRHO2 /tmp/RRHO2
cp -r /tmp/RRHO2/R .
Rscript tests/r_reference/generate_reference.R
python -m pytest tests/ --run-r-comparison        # all 87 tests
```

`--run-r-comparison` turns a missing oracle into a usage error rather than a
silent skip, which is what you want in CI.

The comparison tolerances are named constants at the top of
`tests/test_against_r.py` (`LOGSF_ATOL`, `LOGSF_RTOL`), each justified above.
