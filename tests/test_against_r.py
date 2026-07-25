"""Compare the Python port against ground truth captured from the R package.

These tests are **optional**: they need the upstream R sources, which are not
distributed with this package for licensing and standalone-packaging reasons.
See "Validating against the R implementation" in README.md for how to fetch them
and generate the ground truth. Without it, every test in this module skips (and
the terminal summary says so).

Each case here corresponds to one entry in
``tests/r_reference/generate_reference.R``.
"""

import csv

import numpy as np
import pytest

from rrho2 import default_step_size, numeric_list_overlap, rrho2
from rrho2._multitest import legacy_adjust_neglog_pvalues

from .conftest import read_gene_list, read_matrix

pytestmark = pytest.mark.requires_r_reference

# name -> keyword arguments matching the R call in generate_reference.R
CASES = {
    "base": {},
    "log10": {"log10": True},
    "step50_boundary25": {"stepsize": 50, "boundary": 0.25},
    "fisher": {"method": "fisher"},
    "shuffled_odd": {"log10": True},
    "ties": {},
    "strong": {},
    # R round-trips p.adjust through the linear scale; reproduce that here so
    # the comparison is exact. The log-space default is checked separately.
    "bh_small": {"multiple_testing": "BH", "log_space_padjust": False},
    "by_small": {"multiple_testing": "BY", "log_space_padjust": False},
}

QUADRANTS = ("uu", "dd", "ud", "du")


def _run(case, **kwargs):
    return rrho2(case.list1, case.list2, **kwargs)


# scipy's hypergeom.logsf has an absolute error floor of ~4e-12 relative to
# exact arithmetic, where R's phyper is accurate to ~1e-15 relative. atol
# absorbs that floor; rtol is then tight enough to catch any real algorithmic
# difference on the significant cells that RRHO exists to find. See
# docs/PORTING_NOTES.md.
LOGSF_ATOL = 1e-10
LOGSF_RTOL = 1e-12


def _assert_matrix_equal(actual, expected, name, rtol=LOGSF_RTOL, atol=LOGSF_ATOL):
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    actual_nan, expected_nan = np.isnan(actual), np.isnan(expected)
    np.testing.assert_array_equal(
        actual_nan, expected_nan, err_msg=f"{name}: separator strip differs"
    )
    a, e = actual[~actual_nan], expected[~expected_nan]
    np.testing.assert_allclose(a, e, rtol=rtol, atol=atol, err_msg=f"{name}: values differ")


@pytest.mark.parametrize("name", list(CASES))
def test_hypermat_matches_r(reference, name):
    case = reference(name)
    result = _run(case, **CASES[name])
    _assert_matrix_equal(result.hypermat, case.hypermat, name)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("quadrant", QUADRANTS)
def test_genelists_match_r(reference, name, quadrant):
    case = reference(name)
    result = _run(case, **CASES[name])
    expected = case.genelist(quadrant)
    actual = result.genelist(quadrant)
    for key in ("list1", "list2", "overlap"):
        got = np.asarray(actual.__getattribute__(key), dtype=object)
        want = expected[key]
        assert len(got) == len(want), (
            f"{name}/{quadrant}/{key}: {len(got)} genes != {len(want)}"
        )
        # Order matters: R's intersect() keeps the order of its first argument.
        np.testing.assert_array_equal(got, want, err_msg=f"{name}/{quadrant}/{key}")


def test_numeric_list_overlap_matches_r(data_dir):
    sample1 = read_gene_list(data_dir / "overlap_sample1.txt")
    sample2 = read_gene_list(data_dir / "overlap_sample2.txt")
    for method in ("hyper", "fisher"):
        result = numeric_list_overlap(sample1, sample2, 17, method=method)
        _assert_matrix_equal(
            result["log_pval"],
            read_matrix(data_dir / f"overlap_{method}_logpval.tsv"),
            f"overlap/{method}/log_pval",
        )
        np.testing.assert_array_equal(
            result["counts"],
            read_matrix(data_dir / f"overlap_{method}_counts.tsv").astype(np.int64),
            err_msg=f"overlap/{method}/counts",
        )


def test_default_step_size_matches_r(data_dir):
    with (data_dir / "default_stepsize.csv").open() as handle:
        for row in csv.DictReader(handle):
            got = default_step_size(int(row["n1"]), int(row["n2"]))
            assert got == int(row["step"]), (row["n1"], row["n2"], got, row["step"])


def test_significant_cells_match_r_to_near_machine_precision(reference):
    """The cells RRHO exists to find agree with R to ~13 significant digits.

    scipy's absolute error floor of ~4e-12 only shows up on cells that are not
    significant, so a tight *relative* tolerance is justified once those are
    excluded.
    """
    # The 4e-12 absolute floor becomes a 4e-14 relative error once -log(p) > 100.
    for name in ("base", "strong"):
        case = reference(name)
        actual = _run(case, **CASES[name]).hypermat
        expected = case.hypermat
        significant = ~np.isnan(expected) & (expected > 100.0)
        assert significant.sum() > 100, name
        np.testing.assert_allclose(
            actual[significant],
            expected[significant],
            rtol=1.5e-13,
            atol=0.0,
            err_msg=f"{name}: significant cells differ",
        )


def test_legacy_padjust_reproduces_r_underflow_to_inf(reference):
    """R's BH correction returns Inf on the 'strong' case; the port reproduces it."""
    case = reference("strong_bh_legacy")
    expected = case.hypermat
    assert np.isinf(expected[~np.isnan(expected)]).any(), "reference should contain Inf"

    with pytest.warns(RuntimeWarning, match="Inf was generated"):
        legacy = _run(case, multiple_testing="BH", log_space_padjust=False)
    _assert_matrix_equal(legacy.hypermat, expected, "strong_bh_legacy")


#: -log(p) above which R's exp(-x) round-trip lands in the denormal range and
#: starts shedding mantissa bits (the smallest normal double is ~exp(-708)).
DENORMAL_THRESHOLD = 708.0


def test_log_space_padjust_stays_finite_and_agrees_where_r_does(reference):
    """The default correction removes R's Inf cells without moving the good ones."""
    case = reference("strong_bh_legacy")
    stable = _run(case, multiple_testing="BH").hypermat
    assert np.isfinite(stable[~np.isnan(stable)]).all()

    legacy = case.hypermat  # straight from R, containing Inf
    # Where R's p-value stays a normal double, the two corrections agree.
    safe = np.isfinite(legacy) & (legacy < DENORMAL_THRESHOLD) & ~np.isnan(stable)
    assert safe.sum() > 100
    np.testing.assert_allclose(
        stable[safe], legacy[safe], rtol=1e-10, atol=1e-10,
        err_msg="log-space and R disagree on cells R can represent",
    )

    # The cells R lost to Inf come back as large finite values.
    lost = np.isinf(legacy)
    assert lost.sum() > 0
    assert np.isfinite(stable[lost]).all()
    assert stable[lost].min() > DENORMAL_THRESHOLD


def test_r_padjust_degrades_in_the_denormal_band(reference):
    """R loses precision well before Inf: exp(-x) is denormal for x > ~708.

    This documents *why* log_space_padjust defaults to True. It is a property of
    the R implementation, reproduced by the legacy path, not a port bug.
    """
    case = reference("strong_bh_legacy")
    legacy_from_r = case.hypermat
    band = np.isfinite(legacy_from_r) & (legacy_from_r > DENORMAL_THRESHOLD)
    assert band.sum() > 0, "reference should exercise the denormal band"

    # The port's legacy path reproduces R's degraded values bit-for-bit...
    with pytest.warns(RuntimeWarning, match="Inf was generated"):
        legacy = _run(case, multiple_testing="BH", log_space_padjust=False).hypermat
    np.testing.assert_allclose(
        legacy[band], legacy_from_r[band], rtol=1e-12, atol=1e-12
    )

    # ...and the log-space path visibly departs from them, by far more than the
    # ~4e-12 numerical noise seen elsewhere.
    stable = _run(case, multiple_testing="BH").hypermat
    assert np.abs(stable[band] - legacy_from_r[band]).max() > 1e-6


def test_legacy_padjust_helper_matches_log_space_when_no_underflow():
    rng = np.random.default_rng(0)
    neglog_p = rng.uniform(0.0, 40.0, size=(13, 11))
    for method in ("BH", "BY"):
        np.testing.assert_allclose(
            legacy_adjust_neglog_pvalues(neglog_p, method),
            __import__("rrho2").adjust_neglog_pvalues(neglog_p, method),
            rtol=1e-9,
            atol=1e-9,
        )
