"""
Tests for the sequential Monte Carlo permutation p-value.

The stopping rule has to save computation without changing what the p-value
means, so these tests check both: that it stops when it should, and that the
p-values it returns behave like p-values.
"""

import numpy as np
import pytest

from photonai_projects.project import PhotonaiProject


sequential_p_value = PhotonaiProject.sequential_p_value


class TestStoppingBehaviour:
    """When sampling stops, and what it reports when it does."""

    def test_stops_once_the_budget_is_spent(self):
        # every permutation beats the observed value, so the budget of 20 is
        # spent on the 20th permutation
        permuted = np.ones(1000)
        result = sequential_p_value(observed=0.0, permuted=permuted,
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is True
        assert result['n_perms_used'] == 20
        assert result['p_value'] == pytest.approx(1.0)

    def test_runs_to_completion_when_nothing_exceeds(self):
        permuted = np.zeros(1000)
        result = sequential_p_value(observed=1.0, permuted=permuted,
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is False
        assert result['n_exceedances'] == 0
        assert result['n_perms_used'] == 1000
        # the smallest attainable p-value with 1000 permutations
        assert result['p_value'] == pytest.approx(1 / 1001)

    def test_stopping_point_depends_on_run_order(self):
        # same multiset of results, different order: the run that front-loads
        # the exceedances spends its budget sooner. Observed sits strictly
        # between the two values so the zeros genuinely do not exceed it.
        front_loaded = np.concatenate([np.ones(20), np.zeros(980)])
        back_loaded = np.concatenate([np.zeros(980), np.ones(20)])

        early = sequential_p_value(0.5, front_loaded, True, 20, 1000)
        late = sequential_p_value(0.5, back_loaded, True, 20, 1000)

        assert early['n_perms_used'] == 20
        assert late['n_perms_used'] == 1000
        assert early['p_value'] > late['p_value']

    def test_a_null_analysis_stops_far_short_of_the_budget(self):
        # observed value sits in the middle of the null distribution, so roughly
        # half of the permutations exceed it and the budget goes quickly
        rng = np.random.RandomState(0)
        permuted = rng.normal(size=1000)
        result = sequential_p_value(observed=0.0, permuted=permuted,
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is True
        assert result['n_perms_used'] < 100, "should save most of the budget"


class TestMetricDirection:
    """Lower-is-better metrics have to be handled the other way round."""

    def test_lower_is_better_counts_smaller_values(self):
        permuted = np.zeros(1000)

        # for an error metric, permutations scoring *below* the observed value
        # are the extreme ones
        exceeded = sequential_p_value(observed=1.0, permuted=permuted,
                                      greater_is_better=False,
                                      max_exceedances=20, n_perms=1000)
        assert exceeded['stopped_early'] is True

        not_exceeded = sequential_p_value(observed=-1.0, permuted=permuted,
                                          greater_is_better=False,
                                          max_exceedances=20, n_perms=1000)
        assert not_exceeded['stopped_early'] is False

    def test_ties_count_as_extreme(self):
        # a permutation equalling the observed value is not evidence for the
        # alternative, so it must count towards the budget
        permuted = np.zeros(100)
        result = sequential_p_value(observed=0.0, permuted=permuted,
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is True
        assert result['n_perms_used'] == 20


class TestConservativeHandling:
    """Runs that failed or never happened must not inflate significance."""

    def test_failed_runs_count_as_exceedances(self):
        permuted = np.full(1000, np.nan)
        result = sequential_p_value(observed=1.0, permuted=permuted,
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is True

    def test_planned_but_missing_runs_count_as_exceedances(self):
        # only 100 of 1000 planned permutations completed, none of them extreme
        result = sequential_p_value(observed=1.0, permuted=np.zeros(100),
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is False
        # the 900 that never ran are counted against significance
        assert result['p_value'] == pytest.approx(901 / 1001)

    def test_no_permutations_at_all_gives_no_evidence(self):
        result = sequential_p_value(observed=1.0, permuted=[],
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['p_value'] == pytest.approx(1.0)


class TestValidity:
    """The p-value has to stay a p-value."""

    def test_uniform_under_the_null(self):
        # Under H0 the observed statistic is exchangeable with the permuted
        # ones, so p-values should be roughly uniform and reject at close to
        # the nominal rate rather than above it.
        rng = np.random.RandomState(1)
        alpha = 0.05
        rejections = 0
        n_experiments = 2000

        for _ in range(n_experiments):
            draws = rng.normal(size=201)
            observed, permuted = draws[0], draws[1:]
            result = sequential_p_value(observed, permuted,
                                        greater_is_better=True,
                                        max_exceedances=20, n_perms=200)
            rejections += result['p_value'] <= alpha

        rate = rejections / n_experiments
        # binomial standard error at alpha=0.05 over 2000 draws is ~0.005;
        # allow three of those and require the test not to be anticonservative
        assert rate <= alpha + 0.015, f"rejection rate {rate:.4f} exceeds {alpha}"

    def test_never_more_significant_than_the_exhaustive_estimator(self):
        # stopping early must not manufacture significance: whenever the rule
        # stops, the p-value it reports stays above the smallest attainable one
        rng = np.random.RandomState(2)
        for _ in range(200):
            permuted = rng.normal(size=1000)
            observed = rng.normal()
            result = sequential_p_value(observed, permuted,
                                        greater_is_better=True,
                                        max_exceedances=20, n_perms=1000)
            if result['stopped_early']:
                assert result['p_value'] >= 20 / 1000

    def test_strong_signal_still_reaches_significance(self):
        # the saving must not cost power: an observed value beyond every
        # permutation still gets the smallest possible p-value
        rng = np.random.RandomState(3)
        permuted = rng.normal(size=1000)
        result = sequential_p_value(observed=10.0, permuted=permuted,
                                    greater_is_better=True,
                                    max_exceedances=20, n_perms=1000)

        assert result['stopped_early'] is False
        assert result['p_value'] == pytest.approx(1 / 1001)


class TestArgumentValidation:

    def test_rejects_a_budget_below_one(self):
        with pytest.raises(ValueError, match="max_exceedances"):
            sequential_p_value(0.0, np.zeros(10), True,
                               max_exceedances=0, n_perms=10)


class TestExceedanceBudget:
    """`stop_above_p` and `max_exceedances` are the same rule, two ways."""

    def test_threshold_translates_to_a_count(self):
        assert PhotonaiProject.resolve_exceedance_budget(
            n_perms=1000, stop_above_p=0.1) == 100
        assert PhotonaiProject.resolve_exceedance_budget(
            n_perms=1000, stop_above_p=0.02) == 20

    def test_rounds_up_so_the_threshold_is_never_undershot(self):
        # 0.1 * 333 = 33.3; rounding down would allow stopping at p = 0.099
        budget = PhotonaiProject.resolve_exceedance_budget(
            n_perms=333, stop_above_p=0.1)
        assert budget == 34
        assert budget / 333 >= 0.1

    def test_a_count_passes_through_unchanged(self):
        assert PhotonaiProject.resolve_exceedance_budget(
            n_perms=1000, max_exceedances=42) == 42

    @pytest.mark.parametrize("kwargs", [
        {},                                             # neither
        {"stop_above_p": 0.1, "max_exceedances": 100},  # both
    ])
    def test_requires_exactly_one_of_the_two(self, kwargs):
        with pytest.raises(ValueError, match="exactly one"):
            PhotonaiProject.resolve_exceedance_budget(n_perms=1000, **kwargs)

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_rejects_thresholds_outside_the_unit_interval(self, threshold):
        with pytest.raises(ValueError, match="stop_above_p"):
            PhotonaiProject.resolve_exceedance_budget(
                n_perms=1000, stop_above_p=threshold)


class TestThresholdSemantics:
    """The promise of `stop_above_p`: never stop below the stated p-value."""

    @pytest.mark.parametrize("threshold", [0.05, 0.1, 0.2])
    def test_never_stops_below_the_threshold(self, threshold):
        rng = np.random.RandomState(7)
        n_perms = 1000
        budget = PhotonaiProject.resolve_exceedance_budget(
            n_perms=n_perms, stop_above_p=threshold)

        for _ in range(300):
            true_p = rng.uniform(0.001, 1.0)
            permuted = np.where(rng.rand(n_perms) < true_p, 1.0, -1.0)
            result = sequential_p_value(0.0, permuted, True, budget, n_perms)
            if result['stopped_early']:
                assert result['p_value'] >= threshold

    def test_analyses_below_the_threshold_use_the_full_budget(self):
        rng = np.random.RandomState(8)
        n_perms = 1000
        budget = PhotonaiProject.resolve_exceedance_budget(
            n_perms=n_perms, stop_above_p=0.1)

        # true p well below 0.1: must never stop early
        for _ in range(100):
            permuted = np.where(rng.rand(n_perms) < 0.01, 1.0, -1.0)
            result = sequential_p_value(0.0, permuted, True, budget, n_perms)
            assert result['stopped_early'] is False
            assert result['n_perms_used'] == n_perms
