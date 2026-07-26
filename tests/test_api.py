"""Behaviour of the Python API: input handling, validation, and the new options."""

import numpy as np
import pytest

from rrho2 import QUADRANTS, default_step_size, rrho2


def make_lists(n_genes=600, n_de=60, seed=0):
    """Concordant synthetic lists in the shape the package documents."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"Gene{i}" for i in range(n_genes)], dtype=object)
    n_noise = n_genes - 2 * n_de

    def scores():
        up = -np.log10(rng.uniform(0, 0.05, n_de))
        down = np.log10(rng.uniform(0, 0.05, n_de))
        noise = -np.log10(rng.uniform(0, 1, n_noise)) * rng.choice([1, -1], n_noise)
        return np.concatenate([up, down, noise])

    return (genes, scores()), (genes, scores())


@pytest.fixture(scope="module")
def lists():
    return make_lists()


# --------------------------------------------------------------------------
# Input coercion
# --------------------------------------------------------------------------


def test_accepts_names_values_pair(lists):
    l1, l2 = lists
    result = rrho2(l1, l2)
    assert result.hypermat.ndim == 2


def test_accepts_n_by_2_array(lists):
    l1, l2 = lists
    a1 = np.column_stack([l1[0], l1[1]]).astype(object)
    a2 = np.column_stack([l2[0], l2[1]]).astype(object)
    np.testing.assert_array_equal(rrho2(a1, a2).hypermat, rrho2(l1, l2).hypermat)


def test_accepts_pandas_dataframe(lists):
    pd = pytest.importorskip("pandas")
    l1, l2 = lists
    d1 = pd.DataFrame({"Genes": l1[0], "DDE": l1[1]})
    d2 = pd.DataFrame({"Genes": l2[0], "DDE": l2[1]})
    np.testing.assert_array_equal(rrho2(d1, d2).hypermat, rrho2(l1, l2).hypermat)


def test_gene_order_does_not_matter(lists):
    """Both lists are sorted internally, so input order is irrelevant."""
    l1, l2 = lists
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(l2[0]))
    shuffled = (l2[0][perm], l2[1][perm])
    baseline = rrho2(l1, l2)
    reordered = rrho2(l1, shuffled)
    np.testing.assert_allclose(reordered.hypermat, baseline.hypermat, equal_nan=True)
    for quadrant in QUADRANTS:
        np.testing.assert_array_equal(
            reordered.genelist(quadrant).overlap, baseline.genelist(quadrant).overlap
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rejects_duplicate_identifiers(lists):
    l1, l2 = lists
    dup_names = l1[0].copy()
    dup_names[1] = dup_names[0]
    with pytest.raises(ValueError, match="Non-unique gene identifier.*list1"):
        rrho2((dup_names, l1[1]), l2)


def test_rejects_nan_scores(lists):
    l1, l2 = lists
    values = l1[1].copy()
    values[3] = np.nan
    with pytest.raises(ValueError, match="NA value exists in list1"):
        rrho2((l1[0], values), l2)


def test_nan_error_mentions_the_opt_out(lists):
    l1, l2 = lists
    values = l1[1].copy()
    values[[3, 9]] = np.nan
    with pytest.raises(ValueError, match=r"\(2 of \d+\).*drop_nan=True"):
        rrho2((l1[0], values), l2)


# --------------------------------------------------------------------------
# drop_nan
# --------------------------------------------------------------------------


def test_drop_nan_equals_pre_filtering_by_hand(lists):
    """The central guarantee: dropping is equivalent to never passing the gene.

    This is what keeps the p-values calibrated -- the hypergeometric population
    becomes the reduced size, rather than the map being built on n genes with
    holes punched in it.
    """
    l1, l2 = lists
    names, values1 = l1
    values2 = l2[1]
    missing = [2, 17, 40]

    with_nan = values1.copy()
    with_nan[missing] = np.nan
    dropped = rrho2((names, with_nan), (names, values2), drop_nan=True)

    keep = np.ones(len(names), dtype=bool)
    keep[missing] = False
    manual = rrho2((names[keep], values1[keep]), (names[keep], values2[keep]))

    np.testing.assert_allclose(dropped.hypermat, manual.hypermat, equal_nan=True)
    assert dropped.stepsize == manual.stepsize
    assert dropped.n_genes == manual.n_genes == len(names) - len(missing)
    assert dropped.n_dropped == len(missing)
    for quadrant in QUADRANTS:
        np.testing.assert_array_equal(
            dropped.genelist(quadrant).overlap, manual.genelist(quadrant).overlap
        )


def test_drop_nan_removes_gene_from_both_lists(lists):
    """A gene missing in one list cannot be ranked in the other either."""
    l1, l2 = lists
    names = l1[0]
    with_nan = l1[1].copy()
    with_nan[[5, 6]] = np.nan
    gone = set(names[[5, 6]].tolist())

    result = rrho2((names, with_nan), (names, l2[1]), drop_nan=True)
    assert result.n_dropped == 2
    for quadrant in QUADRANTS:
        genes = result.genelist(quadrant)
        assert not (set(genes.list1) & gone)
        assert not (set(genes.list2) & gone), "dropped from list1 only"


def test_drop_nan_takes_the_union_across_both_lists(lists):
    l1, l2 = lists
    names = l1[0]
    a = l1[1].copy()
    b = l2[1].copy()
    a[[1, 2, 3]] = np.nan
    b[[3, 4]] = np.nan  # index 3 overlaps, so the union is {1, 2, 3, 4}

    result = rrho2((names, a), (names, b), drop_nan=True)
    assert result.n_dropped == 4
    assert result.n_genes == len(names) - 4


def test_drop_nan_matches_by_name_not_position(lists):
    """The lists need not be in the same order, so dropping must key on names."""
    l1, l2 = lists
    names = l1[0]
    with_nan = l1[1].copy()
    with_nan[[7, 8]] = np.nan

    rng = np.random.default_rng(2)
    perm = rng.permutation(len(names))
    shuffled = rrho2(
        (names, with_nan), (names[perm], l2[1][perm]), drop_nan=True
    )
    ordered = rrho2((names, with_nan), (names, l2[1]), drop_nan=True)

    assert shuffled.n_dropped == ordered.n_dropped == 2
    np.testing.assert_allclose(shuffled.hypermat, ordered.hypermat, equal_nan=True)


def test_drop_nan_is_a_no_op_without_nans(lists):
    l1, l2 = lists
    baseline = rrho2(l1, l2)
    with_flag = rrho2(l1, l2, drop_nan=True)
    np.testing.assert_array_equal(with_flag.hypermat, baseline.hypermat)
    assert with_flag.n_dropped == 0
    assert with_flag.n_genes == len(l1[0])


def test_drop_nan_reports_counts_by_default(lists):
    """n_genes/n_dropped are populated even when drop_nan is not used."""
    l1, l2 = lists
    result = rrho2(l1, l2)
    assert result.n_genes == len(l1[0])
    assert result.n_dropped == 0


def test_drop_nan_rederives_default_stepsize(lists):
    """A smaller surviving list means a finer default grid, not the original one."""
    l1, l2 = lists
    names, values1 = l1
    n = len(names)
    with_nan = values1.copy()
    with_nan[: n // 4] = np.nan
    surviving = n - n // 4

    result = rrho2((names, with_nan), (names, l2[1]), drop_nan=True)
    assert result.n_genes == surviving
    assert result.stepsize == default_step_size(surviving, surviving)
    # An explicit stepsize still wins.
    pinned = rrho2((names, with_nan), (names, l2[1]), drop_nan=True, stepsize=25)
    assert pinned.stepsize == 25


def test_drop_nan_rejects_all_missing(lists):
    l1, l2 = lists
    with pytest.raises(ValueError, match="nothing is left to compare"):
        rrho2((l1[0], np.full(len(l1[0]), np.nan)), l2, drop_nan=True)


def test_drop_nan_still_enforces_other_validation(lists):
    """Duplicate identifiers remain an error regardless of drop_nan."""
    l1, l2 = lists
    dup = l1[0].copy()
    dup[1] = dup[0]
    with pytest.raises(ValueError, match="Non-unique gene identifier"):
        rrho2((dup, l1[1]), l2, drop_nan=True)


def test_drop_nan_composes_with_intersection(lists):
    """NaN dropping and the shared-gene restriction stack correctly."""
    l1, l2 = lists
    names, values1 = l1
    values2 = l2[1].copy()

    with_nan = values1.copy()
    with_nan[[0, 1]] = np.nan       # dropped for being missing
    trimmed_names = names[:-3]      # last 3 genes absent from list2
    trimmed_values = values2[:-3]

    result = rrho2(
        (names, with_nan), (trimmed_names, trimmed_values), drop_nan=True
    )
    assert result.n_dropped == 2
    assert result.n_unshared == 3
    assert result.n_genes == len(names) - 5

    # Equivalent to filtering both out by hand.
    keep = np.ones(len(names), dtype=bool)
    keep[[0, 1]] = False
    keep[-3:] = False
    manual = rrho2((names[keep], values1[keep]), (names[keep], values2[keep]))
    np.testing.assert_allclose(result.hypermat, manual.hypermat, equal_nan=True)


def test_drop_nan_with_pandas_na(lists):
    pd = pytest.importorskip("pandas")
    l1, l2 = lists
    frame1 = pd.DataFrame({"Genes": l1[0], "DDE": l1[1]})
    frame1.loc[[3, 11], "DDE"] = np.nan
    frame2 = pd.DataFrame({"Genes": l2[0], "DDE": l2[1]})
    result = rrho2(frame1, frame2, drop_nan=True)
    assert result.n_dropped == 2
    assert result.n_genes == len(l1[0]) - 2


def test_mismatched_gene_sets_are_intersected(lists):
    """Genes present in only one list are dropped, not an error."""
    l1, l2 = lists
    other = l2[0].copy()
    other[0] = "NotInList1"
    result = rrho2(l1, (other, l2[1]))
    # One gene left list2's set and one arrived, so two are unshared.
    assert result.n_unshared == 2
    assert result.n_genes == len(l1[0]) - 1


def test_rejects_bad_arguments(lists):
    l1, l2 = lists
    with pytest.raises(ValueError, match="method"):
        rrho2(l1, l2, method="nope")
    with pytest.raises(ValueError, match="multiple_testing"):
        rrho2(l1, l2, multiple_testing="bonferroni")
    with pytest.raises(ValueError, match="boundary"):
        rrho2(l1, l2, boundary=1.5)
    with pytest.raises(ValueError, match="labels"):
        rrho2(l1, l2, labels=("only-one",))
    with pytest.raises(ValueError, match="not p-values"):
        rrho2(l1, l2, method="fisher", multiple_testing="BH")


def test_single_signed_scores_warn_and_build_partial_map(lists):
    """A single-signed list has no direction split, so two quadrants are empty.

    R indexes with ``(boundary+1):len``, which counts backwards rather than being
    empty, and produces a malformed map. The port builds the quadrants that do
    exist and warns.
    """
    l1, l2 = lists
    all_positive = np.abs(l1[1])

    with pytest.warns(RuntimeWarning, match="no down-regulated block"):
        result = rrho2((l1[0], all_positive), l2)
    # list1 has no down block, so dd and du cannot exist.
    assert result.genelist_dd.peak is None
    assert result.genelist_du.peak is None
    assert result.genelist_uu.peak is not None
    assert result.genelist_ud.peak is not None

    with pytest.warns(RuntimeWarning, match="no up-regulated block"):
        flipped = rrho2((l1[0], -all_positive), l2)
    assert flipped.genelist_uu.peak is None
    assert flipped.genelist_ud.peak is None
    assert flipped.genelist_dd.peak is not None
    assert flipped.genelist_du.peak is not None


# --------------------------------------------------------------------------
# Shared-gene intersection
# --------------------------------------------------------------------------


def test_intersection_equals_pre_filtering_by_hand(lists):
    """Restricting to shared genes matches never passing the extras."""
    l1, l2 = lists
    names, values1 = l1
    values2 = l2[1]

    # list2 loses the first 20 genes and gains 12 that list1 never had.
    keep = np.ones(len(names), dtype=bool)
    keep[:20] = False
    rng = np.random.default_rng(4)
    extra_names = np.array([f"Only2_{i}" for i in range(12)], dtype=object)
    extra_values = rng.normal(size=12)
    names2 = np.concatenate([names[keep], extra_names])
    values2b = np.concatenate([values2[keep], extra_values])

    result = rrho2((names, values1), (names2, values2b))
    assert result.n_unshared == 20 + 12
    assert result.n_genes == len(names) - 20

    manual = rrho2((names[keep], values1[keep]), (names[keep], values2[keep]))
    np.testing.assert_allclose(result.hypermat, manual.hypermat, equal_nan=True)
    assert result.stepsize == manual.stepsize
    for quadrant in QUADRANTS:
        np.testing.assert_array_equal(
            result.genelist(quadrant).overlap, manual.genelist(quadrant).overlap
        )


def test_intersection_never_reports_unshared_genes(lists):
    l1, l2 = lists
    names = l1[0]
    extra = np.array(["Ghost1", "Ghost2"], dtype=object)
    names2 = np.concatenate([names, extra])
    values2 = np.concatenate([l2[1], [5.0, -5.0]])

    result = rrho2(l1, (names2, values2))
    assert result.n_unshared == 2
    for quadrant in QUADRANTS:
        genes = result.genelist(quadrant)
        assert not (set(genes.list1) & set(extra.tolist()))
        assert not (set(genes.list2) & set(extra.tolist()))


def test_identical_lists_report_no_unshared(lists):
    l1, l2 = lists
    result = rrho2(l1, l2)
    assert result.n_unshared == 0
    assert result.n_genes == len(l1[0])


def test_intersection_is_order_independent(lists):
    l1, l2 = lists
    names = l1[0]
    keep = np.ones(len(names), dtype=bool)
    keep[5:15] = False
    rng = np.random.default_rng(6)
    perm = rng.permutation(int(keep.sum()))

    ordered = rrho2(l1, (names[keep], l2[1][keep]))
    shuffled = rrho2(l1, (names[keep][perm], l2[1][keep][perm]))
    np.testing.assert_allclose(ordered.hypermat, shuffled.hypermat, equal_nan=True)
    assert ordered.n_unshared == shuffled.n_unshared == 10


def test_disjoint_gene_sets_raise(lists):
    l1, l2 = lists
    other = np.array([f"Other{i}" for i in range(len(l1[0]))], dtype=object)
    with pytest.raises(ValueError, match="no gene identifiers in common"):
        rrho2(l1, (other, l2[1]))


# --------------------------------------------------------------------------
# Single-signed lists
# --------------------------------------------------------------------------


def test_strictly_positive_lists_yield_one_populated_quadrant(lists):
    """The exact degenerate edge: boundary == len, so only uu can exist."""
    l1, _ = lists
    names = l1[0]
    rng = np.random.default_rng(8)
    a = rng.uniform(0.5, 10.0, len(names))
    b = rng.uniform(0.5, 10.0, len(names))

    with pytest.warns(RuntimeWarning, match="no down-regulated block"):
        result = rrho2((names, a), (names, b))

    n_steps = len(np.arange(1, len(names) + 1, result.stepsize))
    assert result.boundary1 == result.boundary2 == n_steps

    # uu covers the whole grid; the other three are empty.
    assert result.genelist_uu.peak is not None
    assert len(result.genelist_uu.overlap) > 0
    for quadrant in ("dd", "ud", "du"):
        genes = result.genelist(quadrant)
        assert genes.peak is None
        assert genes.sizes == (0, 0, 0)

    populated = ~np.isnan(result.hypermat)
    assert populated.sum() == n_steps * n_steps


def test_strictly_negative_lists_yield_only_dd(lists):
    l1, _ = lists
    names = l1[0]
    rng = np.random.default_rng(10)
    a = -rng.uniform(0.5, 10.0, len(names))
    b = -rng.uniform(0.5, 10.0, len(names))

    with pytest.warns(RuntimeWarning, match="no up-regulated block"):
        result = rrho2((names, a), (names, b))

    assert result.boundary1 == result.boundary2 == 0
    assert result.genelist_dd.peak is not None
    assert len(result.genelist_dd.overlap) > 0
    for quadrant in ("uu", "ud", "du"):
        assert result.genelist(quadrant).peak is None


def test_one_single_signed_list_keeps_two_quadrants(lists):
    """Only the offending list loses its block; the other still splits."""
    l1, l2 = lists
    with pytest.warns(RuntimeWarning, match="list1 has no down-regulated block"):
        result = rrho2((l1[0], np.abs(l1[1])), l2)

    # list2 still has both directions, so uu and ud survive.
    assert result.genelist_uu.peak is not None
    assert result.genelist_ud.peak is not None
    assert result.genelist_dd.peak is None
    assert result.genelist_du.peak is None
    assert 0 < result.boundary2 < len(
        np.arange(1, result.n_genes + 1, result.stepsize)
    )


def test_single_signed_map_still_plots(lists):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    l1, l2 = lists
    with pytest.warns(RuntimeWarning):
        result = rrho2((l1[0], np.abs(l1[1])), l2, labels=("a", "b"))
    result.heatmap()
    for quadrant in QUADRANTS:
        result.venn(quadrant)  # including the empty ones
    plt.close("all")


def test_normal_signed_data_emits_no_warning(lists):
    """The warning must not fire for well-formed input."""
    import warnings

    l1, l2 = lists
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rrho2(l1, l2)


# --------------------------------------------------------------------------
# Structure and options
# --------------------------------------------------------------------------


def test_separator_strip_is_nan_and_correctly_sized(lists):
    l1, l2 = lists
    result = rrho2(l1, l2, boundary=0.2)
    rows, cols = result.hypermat.shape
    assert rows == result.strip1 + (rows - result.strip1)
    strip_rows = slice(result.boundary1, result.boundary1 + result.strip1)
    strip_cols = slice(result.boundary2, result.boundary2 + result.strip2)
    assert np.isnan(result.hypermat[strip_rows, :]).all()
    assert np.isnan(result.hypermat[:, strip_cols]).all()
    # Everything outside the cross-shaped strip is populated.
    outside = ~np.isnan(result.hypermat)
    assert outside.sum() == (rows - result.strip1) * (cols - result.strip2)


def test_concordant_data_peaks_on_the_diagonal_quadrants(lists):
    """uu and dd should dominate ud and du for concordant lists."""
    l1, l2 = lists
    result = rrho2(l1, l2, log10=True)
    hm = result.hypermat

    def quadrant_max(rows, cols):
        return np.nanmax(hm[rows, cols])

    up1, up2 = slice(0, result.boundary1), slice(0, result.boundary2)
    down1 = slice(result.strip1 + result.boundary1, hm.shape[0])
    down2 = slice(result.strip2 + result.boundary2, hm.shape[1])
    concordant = min(quadrant_max(up1, up2), quadrant_max(down1, down2))
    discordant = max(quadrant_max(up1, down2), quadrant_max(down1, up2))
    assert concordant > discordant


def test_log10_is_a_pure_rescaling(lists):
    l1, l2 = lists
    natural = rrho2(l1, l2).hypermat
    base10 = rrho2(l1, l2, log10=True).hypermat
    np.testing.assert_allclose(base10, natural * np.log10(np.e), equal_nan=True)


def test_genelists_are_consistent_with_their_quadrant(lists):
    l1, l2 = lists
    result = rrho2(l1, l2)
    all_genes = set(l1[0].tolist())
    for quadrant in QUADRANTS:
        genes = result.genelist(quadrant)
        assert set(genes.list1) <= all_genes
        assert set(genes.list2) <= all_genes
        # overlap is exactly the intersection, in list1 order
        assert set(genes.overlap) == set(genes.list1) & set(genes.list2)
        assert len(genes.overlap) == len(set(genes.overlap))
        order = {g: i for i, g in enumerate(genes.list1)}
        positions = [order[g] for g in genes.overlap]
        assert positions == sorted(positions)


def test_population_offset_zero_changes_values_slightly(lists):
    """The R default models an urn with one extra element; 0 is correct.

    The correction is worth a fraction of a percent, so it does not change any
    interpretation -- but it is a real difference, hence opt-in.
    """
    l1, l2 = lists
    r_compatible = rrho2(l1, l2, population_offset=1).hypermat
    corrected = rrho2(l1, l2, population_offset=0).hypermat
    finite = ~np.isnan(r_compatible)
    assert not np.allclose(r_compatible[finite], corrected[finite])
    # Same order of magnitude everywhere; the peak moves by well under 1%.
    peak_shift = abs(np.nanmax(corrected) - np.nanmax(r_compatible)) / np.nanmax(
        r_compatible
    )
    assert peak_shift < 0.01


def test_population_offset_zero_matches_textbook_hypergeometric(lists):
    """With offset 0, a single cell agrees with a direct scipy.stats call."""
    from scipy.stats import hypergeom

    l1, l2 = lists
    n = len(l1[0])
    result = rrho2(l1, l2, population_offset=0, return_counts=True)
    counts = result.counts
    prefix = np.arange(1, n + 1, result.stepsize)
    i, j = 5, 7
    a, b, c = prefix[i], prefix[j], counts[i, j]
    expected = -hypergeom.logsf(c - 1, n, a, b)  # population = n exactly
    # Read the same cell out of the un-split uu quadrant.
    assert i < result.boundary1 and j < result.boundary2
    np.testing.assert_allclose(result.hypermat[i, j], expected, rtol=1e-12)


def test_return_counts_matches_direct_set_intersections(lists):
    """The cumsum shortcut agrees with brute-force set intersection."""
    l1, l2 = lists
    result = rrho2(l1, l2, return_counts=True)
    names1 = l1[0][np.argsort(-l1[1], kind="stable")]
    names2 = l2[0][np.argsort(-l2[1], kind="stable")]
    prefix = np.arange(1, len(names1) + 1, result.stepsize)
    for i in (0, 3, len(prefix) // 2, len(prefix) - 1):
        for j in (0, 5, len(prefix) - 1):
            brute = len(set(names1[: prefix[i]]) & set(names2[: prefix[j]]))
            assert result.counts[i, j] == brute, (i, j)


def test_stepsize_default_and_override(lists):
    l1, l2 = lists
    n = len(l1[0])
    assert rrho2(l1, l2).stepsize == default_step_size(n, n)
    assert rrho2(l1, l2, stepsize=40).stepsize == 40


def test_fisher_produces_log_odds(lists):
    l1, l2 = lists
    result = rrho2(l1, l2, method="fisher")
    assert result.method == "fisher"
    # Concordant lists give positive log odds at the uu peak.
    assert np.nanmax(result.hypermat[: result.boundary1, : result.boundary2]) > 0


def test_boundary_zero_gives_no_strip(lists):
    l1, l2 = lists
    result = rrho2(l1, l2, boundary=0.0)
    assert result.strip1 == 0 and result.strip2 == 0
    assert not np.isnan(result.hypermat).any()


# --------------------------------------------------------------------------
# Array access for custom plotting
# --------------------------------------------------------------------------


def test_quadrant_map_matches_manual_slicing(lists):
    l1, l2 = lists
    result = rrho2(l1, l2, log10=True)
    hm = result.hypermat
    expected = {
        "uu": hm[: result.boundary1, : result.boundary2],
        "dd": hm[result.strip1 + result.boundary1 :, result.strip2 + result.boundary2 :],
        "ud": hm[: result.boundary1, result.strip2 + result.boundary2 :],
        "du": hm[result.strip1 + result.boundary1 :, : result.boundary2],
    }
    for quadrant, want in expected.items():
        np.testing.assert_array_equal(result.quadrant_map(quadrant), want)


def test_quadrant_map_excludes_the_nan_strip(lists):
    """A quadrant pulled out on its own must be fully populated."""
    l1, l2 = lists
    result = rrho2(l1, l2, boundary=0.25)
    for quadrant in QUADRANTS:
        block = result.quadrant_map(quadrant)
        assert block.size > 0
        assert not np.isnan(block).any(), quadrant
    # The four quadrants tile every non-nan cell exactly once.
    total = sum(result.quadrant_map(q).size for q in QUADRANTS)
    assert total == int(np.sum(~np.isnan(result.hypermat)))


def test_rank_cutoffs_agree_with_genelist_sizes(lists):
    """The axis labels must be the cutoffs the gene lists were read from.

    This pins the down-direction case, where a cutoff is a suffix length rather
    than a prefix length.
    """
    l1, l2 = lists
    result = rrho2(l1, l2, log10=True)
    for quadrant in QUADRANTS:
        rows, cols = result.quadrant_slices(quadrant)
        cutoffs1, cutoffs2 = result.rank_cutoffs(quadrant)
        genes = result.genelist(quadrant)
        block = result.quadrant_map(quadrant)
        assert len(cutoffs1) == block.shape[0]
        assert len(cutoffs2) == block.shape[1]

        row, col = genes.peak
        i, j = row - rows.start, col - cols.start
        assert cutoffs1[i] == len(genes.list1), quadrant
        assert cutoffs2[j] == len(genes.list2), quadrant


def test_rank_cutoffs_are_valid_list_lengths(lists):
    l1, l2 = lists
    result = rrho2(l1, l2)
    for quadrant in QUADRANTS:
        for axis in result.rank_cutoffs(quadrant):
            assert axis.min() >= 1
            assert axis.max() <= result.n_genes


def test_quadrant_peaks_match_nanmax_of_each_block(lists):
    l1, l2 = lists
    result = rrho2(l1, l2, log10=True)
    peaks = result.quadrant_peaks()
    assert set(peaks) == set(QUADRANTS)
    for quadrant, value in peaks.items():
        assert value == pytest.approx(np.nanmax(result.quadrant_map(quadrant)))
    # Concordant data: uu/dd dominate ud/du.
    assert min(peaks["uu"], peaks["dd"]) > max(peaks["ud"], peaks["du"])


def test_quadrant_accessors_handle_empty_quadrants(lists):
    l1, l2 = lists
    with pytest.warns(RuntimeWarning):
        result = rrho2((l1[0], np.abs(l1[1])), l2)
    peaks = result.quadrant_peaks()
    for quadrant in ("dd", "du"):
        assert result.quadrant_map(quadrant).size == 0
        assert np.isnan(peaks[quadrant])
    for quadrant in ("uu", "ud"):
        assert result.quadrant_map(quadrant).size > 0
        assert np.isfinite(peaks[quadrant])


def test_quadrant_accessors_reject_bad_names(lists):
    l1, l2 = lists
    result = rrho2(l1, l2)
    for method in (result.quadrant_map, result.quadrant_slices, result.rank_cutoffs):
        with pytest.raises(ValueError, match="quadrant"):
            method("xx")


# --------------------------------------------------------------------------
# Plotting smoke tests
# --------------------------------------------------------------------------


def test_heatmap_and_venn_render(lists):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    l1, l2 = lists
    result = rrho2(l1, l2, labels=("list1", "list2"), log10=True)
    ax, image = result.heatmap()
    assert image.get_array().shape == result.hypermat.T.shape
    plt.close("all")

    for quadrant in QUADRANTS:
        result.venn(quadrant)
    plt.close("all")

    with pytest.raises(ValueError, match="minimum > maximum"):
        result.heatmap(minimum=10, maximum=1)
    plt.close("all")

    with pytest.raises(ValueError, match="quadrant"):
        result.venn("xx")


# --------------------------------------------------------------------------
# log_ranks: geometrically spaced rank cutoffs
# --------------------------------------------------------------------------


def test_log_prefixes_properties():
    from rrho2._overlap import log_prefixes

    for n in (1, 2, 5, 50, 2000, 20000):
        p = log_prefixes(n)
        assert p[0] == 1, n
        assert p[-1] <= n, n
        assert (np.diff(p) > 0).all(), n            # strictly increasing
        assert p.dtype == np.int64
    # Dense at the top, coarse in the tail.
    p = log_prefixes(20000)
    assert np.diff(p)[0] == 1
    assert np.diff(p)[-1] > 100
    assert p[-1] == 20000


def test_log_prefixes_rejects_bad_input():
    from rrho2._overlap import log_prefixes

    with pytest.raises(ValueError, match="n must be at least 1"):
        log_prefixes(0)
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        log_prefixes(100, n_bins=0)


def test_log_ranks_concentrates_resolution_at_the_top(lists):
    l1, l2 = lists
    linear = rrho2(l1, l2, log10=True)
    logged = rrho2(l1, l2, log10=True, log_ranks=True)
    assert logged.log_ranks and not linear.log_ranks

    lin_cut, _ = linear.rank_cutoffs("uu")
    log_cut, _ = logged.rank_cutoffs("uu")
    top = max(1, linear.n_genes // 8)
    assert (log_cut <= top).sum() > (lin_cut <= top).sum()
    # Costs no more than the linear grid.
    assert logged.hypermat.size <= linear.hypermat.size * 1.05


def test_log_ranks_cutoffs_still_match_genelists(lists):
    """The invariant that caught the earlier suffix bug, now on a log grid."""
    l1, l2 = lists
    result = rrho2(l1, l2, log10=True, log_ranks=True)
    for quadrant in QUADRANTS:
        rows, cols = result.quadrant_slices(quadrant)
        cutoffs1, cutoffs2 = result.rank_cutoffs(quadrant)
        genes = result.genelist(quadrant)
        row, col = genes.peak
        assert cutoffs1[row - rows.start] == len(genes.list1), quadrant
        assert cutoffs2[col - cols.start] == len(genes.list2), quadrant


def test_log_ranks_prefixes_are_recorded(lists):
    l1, l2 = lists
    result = rrho2(l1, l2, log_ranks=True)
    p1, p2 = result.prefixes
    assert p1[0] == 1 and (np.diff(p1) > 0).all()
    assert p1[-1] <= result.n_genes
    # rank_cutoffs reads this grid, not stepsize.
    cutoffs1, _ = result.rank_cutoffs("uu")
    np.testing.assert_array_equal(cutoffs1, p1[: len(cutoffs1)])


def test_log_ranks_statistic_matches_a_direct_call_at_shared_cutoffs(lists):
    """Same cutoff must give the same p-value regardless of grid spacing.

    The statistic depends only on (a, b, count, n), so a cutoff appearing in both
    grids must agree. This catches any mis-indexing of the non-uniform grid.
    """
    from scipy.stats import hypergeom

    l1, l2 = lists
    result = rrho2(l1, l2, log_ranks=True, return_counts=True)
    p1, p2 = result.prefixes
    n = result.n_genes
    counts = result.counts
    for i in (0, 3, len(p1) // 2, len(p1) - 1):
        for j in (0, len(p2) // 2, len(p2) - 1):
            expected = -hypergeom.logsf(counts[i, j] - 1, n + 1, p1[i], p2[j])
            # Read the cell out of the uu block where the grid is un-permuted.
            if i < result.boundary1 and j < result.boundary2:
                np.testing.assert_allclose(
                    result.hypermat[i, j], expected, rtol=1e-10, atol=1e-10
                )


def test_log_ranks_finds_the_same_signal(lists):
    """Concordant data stays concordant under log spacing."""
    l1, l2 = lists
    peaks = rrho2(l1, l2, log10=True, log_ranks=True).quadrant_peaks()
    assert min(peaks["uu"], peaks["dd"]) > max(peaks["ud"], peaks["du"])


def test_log_ranks_composes_with_other_options(lists):
    l1, l2 = lists
    for kwargs in (
        {"method": "fisher"},
        {"multiple_testing": "BH"},
        {"boundary": 0.0},
        {"log10": True, "population_offset": 0},
    ):
        result = rrho2(l1, l2, log_ranks=True, **kwargs)
        assert result.log_ranks
        assert np.isfinite(result.hypermat[~np.isnan(result.hypermat)]).all()


def test_log_ranks_plots(lists):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    l1, l2 = lists
    result = rrho2(l1, l2, log10=True, log_ranks=True, labels=("a", "b"))
    result.heatmap()
    result.venn("uu")
    plt.close("all")
