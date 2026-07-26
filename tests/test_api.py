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
    """Duplicates and mismatched gene sets are unaffected by drop_nan."""
    l1, l2 = lists
    dup = l1[0].copy()
    dup[1] = dup[0]
    with pytest.raises(ValueError, match="Non-unique gene identifier"):
        rrho2((dup, l1[1]), l2, drop_nan=True)

    other = l2[0].copy()
    other[0] = "NotInList1"
    with pytest.raises(ValueError, match="must be identical"):
        rrho2(l1, (other, l2[1]), drop_nan=True)


def test_drop_nan_with_pandas_na(lists):
    pd = pytest.importorskip("pandas")
    l1, l2 = lists
    frame1 = pd.DataFrame({"Genes": l1[0], "DDE": l1[1]})
    frame1.loc[[3, 11], "DDE"] = np.nan
    frame2 = pd.DataFrame({"Genes": l2[0], "DDE": l2[1]})
    result = rrho2(frame1, frame2, drop_nan=True)
    assert result.n_dropped == 2
    assert result.n_genes == len(l1[0]) - 2


def test_rejects_mismatched_gene_sets(lists):
    l1, l2 = lists
    other = l2[0].copy()
    other[0] = "NotInList1"
    with pytest.raises(ValueError, match="gene names of the two lists must be identical"):
        rrho2(l1, (other, l2[1]))


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


def test_rejects_single_signed_scores(lists):
    """R would silently build a malformed map; the port refuses instead.

    With no negative scores there is no down-regulated block, and R's
    ``(boundary+1):len`` index counts backwards rather than being empty.
    """
    l1, l2 = lists
    all_positive = np.abs(l1[1])
    with pytest.raises(ValueError, match="no.*down-regulated quadrant"):
        rrho2((l1[0], all_positive), l2)
    with pytest.raises(ValueError, match="no.*up-regulated quadrant"):
        rrho2((l1[0], -all_positive), l2)


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
