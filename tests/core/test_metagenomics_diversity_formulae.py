"""The diversity formulae of metagenomics and the vectors they accept.

Every one of these functions used to take any numbers it was handed.  A negative
count produced a Shannon entropy out of a vector that cannot exist, a NaN quietly
disappeared from the sum, and Simpson's 1-D returned -0.83 — outside the [0, 1]
range its own name promises.  The values here are computed by hand against the
published formulae.
"""
import numpy as np
import pytest

from biosuite.core import metagenomics as mg


# ── known answers ───────────────────────────────────────────────────────────

def test_shannon_entropy_of_a_monoculture_is_zero():
    assert mg.shannon_entropy([7]) == 0.0


def test_shannon_entropy_of_two_equal_taxa_is_ln2():
    assert mg.shannon_entropy([50, 50]) == pytest.approx(np.log(2))
    assert mg.shannon_entropy([2, 2]) == pytest.approx(np.log(2), rel=1e-6)


def test_shannon_entropy_of_four_equal_taxa_is_ln4():
    assert mg.shannon_entropy([50, 50, 50, 50]) == pytest.approx(np.log(4))


def test_shannon_entropy_of_a_peaked_community_is_lower():
    uniform = mg.shannon_entropy([50, 50, 50, 50])
    peaked = mg.shannon_entropy([97, 1, 1, 1])
    assert uniform > peaked
    # -(0.97log0.97 + 3 * 0.01log0.01)
    assert peaked == pytest.approx(
        -(0.97 * np.log(0.97) + 3 * 0.01 * np.log(0.01)))


def test_shannon_ignores_absent_taxa_without_losing_the_rest():
    assert mg.shannon_entropy([0, 4, 0, 4]) == pytest.approx(np.log(2))
    assert mg.shannon_entropy([0, 0, 0]) == 0.0


def test_simpson_index_is_the_unbiased_1_minus_d():
    # 1 - sum n_i(n_i-1) / (N(N-1)) = 1 - 2*(50*49)/(100*99)
    assert mg.simpson_index([50, 50]) == pytest.approx(0.5050505, abs=1e-6)
    # 1 - (30*29 + 20*19 + 15*14 + 35*34) / (100*99) = 1 - 2650/9900
    assert mg.simpson_index([30, 20, 15, 35]) == pytest.approx(0.7323232, abs=1e-6)


def test_simpson_index_of_a_monoculture_is_zero():
    assert mg.simpson_index([4]) == 0.0


def test_simpson_index_of_one_read_is_zero():
    """With one individual the unbiased estimator has nothing to compare, and
    the denominator total*(total-1) would be zero."""
    assert mg.simpson_index([1]) == 0.0
    assert mg.simpson_index([0]) == 0.0


def test_simpson_index_of_an_all_zero_vector_is_zero():
    assert mg.simpson_index([0, 0, 0]) == 0.0


def test_chao1_with_doubletons_uses_the_f1_squared_estimator():
    # S_obs=4, f1=1, f2=2 -> 4 + 1/(2*2) = 4.25
    assert mg.chao1_estimator([1, 2, 2, 5]) == pytest.approx(4.25)
    # S_obs=3, f1=0, f2=1 -> 3
    assert mg.chao1_estimator([5, 4, 2]) == pytest.approx(3)


def test_chao1_without_doubletons_uses_the_bias_corrected_form():
    # S_obs=4, f1=3, f2=0 -> 4 + 3*2/2 = 7
    assert mg.chao1_estimator([1, 1, 1, 5]) == pytest.approx(7)
    # No singleton and no doubleton: nothing to extrapolate from.
    assert mg.chao1_estimator([3, 4, 5]) == pytest.approx(3)


def test_chao1_grows_with_unsampled_rarity():
    without = mg.chao1_estimator([5, 5, 5, 5])
    with_rare = mg.chao1_estimator([5, 5, 1, 1, 1, 1])
    assert with_rare > without


def test_bray_curtis_is_half_between_two_disjoint_halves():
    # sum|a-b| / sum(a+b) = 20/40
    assert mg.bray_curtis_distance([10, 10, 0], [10, 0, 10]) == pytest.approx(0.5)


def test_bray_curtis_of_identical_samples_is_zero_and_symmetric():
    a = np.array([10.0, 0.0, 5.0])
    b = np.array([0.0, 10.0, 5.0])
    assert mg.bray_curtis_distance(a, a) == 0.0
    d = mg.bray_curtis_distance(a, b)
    assert d == pytest.approx(mg.bray_curtis_distance(b, a))
    assert 0 <= d <= 1


def test_bray_curtis_of_completely_disjoint_samples_is_one():
    assert mg.bray_curtis_distance([10, 0], [0, 10]) == pytest.approx(1.0)


def test_bray_curtis_of_two_empty_samples_is_zero():
    assert mg.bray_curtis_distance([0, 0], [0, 0]) == 0.0


def test_the_metrics_return_python_floats_not_numpy_scalars():
    """json.dumps() of the API response, and the equality pins in the API tests,
    both depend on a plain float reaching them."""
    assert type(mg.shannon_entropy([1, 1])) is float
    assert type(mg.simpson_index([1, 1])) is float
    assert type(mg.chao1_estimator([1, 2])) is float
    assert type(mg.bray_curtis_distance([1], [2])) is float


def test_a_zero_entropy_is_zero_not_negative_zero():
    assert repr(mg.shannon_entropy([5])) == "0.0"
    assert repr(mg.shannon_entropy(np.array([0, 0]))) == "0.0"


# ── impossible vectors ──────────────────────────────────────────────────────

@pytest.mark.parametrize("counts", [[-1, 5, 5], [5, -0.5], [-3]])
def test_a_negative_count_is_refused_by_every_metric(counts):
    for fn in (mg.shannon_entropy, mg.simpson_index, mg.chao1_estimator):
        with pytest.raises(ValueError, match="cannot be negative"):
            fn(counts)


@pytest.mark.parametrize("counts", [[float("nan"), 3], [float("inf"), 1],
                                    [1, -float("inf")]])
def test_a_non_finite_count_is_refused_by_every_metric(counts):
    for fn in (mg.shannon_entropy, mg.simpson_index, mg.chao1_estimator):
        with pytest.raises(ValueError, match="must be finite"):
            fn(counts)


@pytest.mark.parametrize("counts", [["a", "b"], [{}], ["1x"]])
def test_counts_that_are_not_numbers_are_refused_in_words(counts):
    for fn in (mg.shannon_entropy, mg.simpson_index):
        with pytest.raises(ValueError, match="must be numbers"):
            fn(counts)


def test_a_none_count_is_refused_as_a_non_finite_one():
    """numpy turns None into nan, so the refusal has to come from the finiteness
    check rather than pretend the value was never a number."""
    with pytest.raises(ValueError, match="must be finite"):
        mg.shannon_entropy([None])


def test_a_two_dimensional_vector_is_refused():
    with pytest.raises(ValueError, match="flat sequence"):
        mg.shannon_entropy([[1, 2], [3, 4]])


def test_mismatched_sample_lengths_are_explained():
    """A raw numpy broadcast error used to be the only answer."""
    with pytest.raises(ValueError, match="same number of taxa, got 2 and 3"):
        mg.bray_curtis_distance([1, 2], [1, 2, 3])


# ── the single-sample entry point ───────────────────────────────────────────

def test_alpha_diversity_single_shannon_matches_the_formula():
    assert mg.alpha_diversity_single(['a', 'a', 'b'], "shannon") == pytest.approx(
        -(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3)))


def test_alpha_diversity_single_simpson_matches_the_formula():
    assert mg.alpha_diversity_single(['a', 'a', 'b'], "simpson") == pytest.approx(
        1 - 2 / 6)


def test_alpha_diversity_single_observed_counts_taxa():
    assert mg.alpha_diversity_single(['a', 'a', 'b', 'c'], "observed") == 3
    assert mg.alpha_diversity_single([], "observed") == 0


def test_alpha_diversity_single_chao1_corner_without_doubletons():
    assert mg.alpha_diversity_single(['a', 'b', 'c'], "chao1") == pytest.approx(6.0)
    assert mg.alpha_diversity_single(['a', 'a', 'b', 'b', 'c'], "chao1") \
        == pytest.approx(3.25)
    assert mg.alpha_diversity_single(['a', 'a'], "chao1") == 1
    assert mg.alpha_diversity_single(['a'] * 3 + ['b'] * 4, "chao1") == 2


def test_alpha_diversity_single_of_an_empty_sample_is_zero():
    for method in mg.DIVERSITY_METHODS:
        assert mg.alpha_diversity_single([], method) == 0.0


def test_alpha_diversity_single_of_one_read_has_no_pair_to_compare():
    """n(n-1) is zero for a single read: the unbiased Simpson estimator has
    nothing to divide, and that is 0.0 rather than a ZeroDivisionError."""
    assert mg.alpha_diversity_single(['a'], "simpson") == 0.0
    assert mg.alpha_diversity_single(['a'], "shannon") == 0.0
    assert mg.alpha_diversity_single(['a'], "observed") == 1
    assert mg.alpha_diversity_single(['a'], "chao1") == 1


@pytest.mark.parametrize("bad_method", ["rasch", "Shannon", "SHANNON", "", "richness",
                                        None, 0])
def test_an_unknown_method_raises_instead_of_answering_zero(bad_method):
    """0.0 is a real diversity value: answering a typo with it told the caller the
    community was a monoculture."""
    with pytest.raises(ValueError, match="method must be one of"):
        mg.alpha_diversity_single(['a', 'b'], bad_method)


def test_the_method_names_the_helper_accepts_are_the_documented_four():
    assert mg.DIVERSITY_METHODS == ('shannon', 'simpson', 'observed', 'chao1')
    for method in mg.DIVERSITY_METHODS:
        assert mg.alpha_diversity_single(['a', 'b'], method) is not None
