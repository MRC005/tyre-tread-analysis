"""Threshold and abstention selection.

The regression these guard against is real and was hit during development: choosing
the lowest threshold that met an 80% worn-recall target produced 0.98 recall at 0.566
balanced accuracy, by calling almost everything worn. A recall target has to be a
constraint on the search, not the thing being maximised.
"""

from __future__ import annotations

import numpy as np
import pytest

from tyretread.models.train import calibration_metrics, choose_threshold


@pytest.fixture
def separable_probabilities(rng):
    """Well-separated out-of-fold probabilities with a realistic overlap region."""
    serviceable = np.clip(rng.beta(2, 6, 200), 0.01, 0.99)
    worn = np.clip(rng.beta(6, 2, 120), 0.01, 0.99)
    probabilities = np.concatenate([serviceable, worn])
    y = np.concatenate([np.zeros(200, dtype=int), np.ones(120, dtype=int)])
    return probabilities, y


def test_meets_the_recall_target_when_one_is_achievable(separable_probabilities):
    probabilities, y = separable_probabilities
    choice = choose_threshold(probabilities, y, target_recall_worn=0.80)
    reaching = [s for s in choice.sweep if s["threshold"] == choice.threshold]
    assert reaching[0]["recall_worn"] >= 0.80


def test_does_not_sacrifice_balanced_accuracy_to_hit_the_target(separable_probabilities):
    """The regression this file exists for."""
    probabilities, y = separable_probabilities
    choice = choose_threshold(probabilities, y, target_recall_worn=0.80)

    qualifying = [s for s in choice.sweep if s["recall_worn"] >= 0.80]
    lowest = min(qualifying, key=lambda s: s["threshold"])
    picked = next(s for s in choice.sweep if s["threshold"] == choice.threshold)

    assert picked["balanced_accuracy"] >= lowest["balanced_accuracy"]
    assert picked["balanced_accuracy"] == max(s["balanced_accuracy"] for s in qualifying)


def test_abstention_rate_respects_its_cap(separable_probabilities):
    probabilities, y = separable_probabilities
    choice = choose_threshold(probabilities, y, max_abstention_rate=0.15)
    assert choice.abstention_rate <= 0.15


def test_a_band_is_only_adopted_if_it_actually_helps(rng):
    """With no overlap to resolve, abstention buys nothing and should not be used."""
    probabilities = np.concatenate([np.full(100, 0.02), np.full(100, 0.98)])
    y = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    choice = choose_threshold(probabilities, y, min_band_gain=0.02)
    assert choice.abstain_band == 0.0
    assert choice.abstention_rate == 0.0


def test_abstention_improves_accuracy_on_the_cases_still_answered(separable_probabilities):
    probabilities, y = separable_probabilities
    choice = choose_threshold(probabilities, y, min_band_gain=0.02, max_abstention_rate=0.40)
    if choice.abstain_band > 0:
        baseline = next(s["balanced_accuracy"] for s in choice.sweep
                        if s["threshold"] == choice.threshold)
        assert choice.balanced_accuracy_on_decided >= baseline + 0.02


def test_an_unreachable_target_is_reported_not_faked(rng):
    """A target that cannot be met must surface as a shortfall, not be silently met."""
    probabilities = rng.uniform(0.45, 0.55, 200)
    y = rng.integers(0, 2, 200)
    choice = choose_threshold(probabilities, y, target_recall_worn=0.999)
    assert choice.target_recall_worn == 0.999
    assert isinstance(choice.recall_worn_on_decided, float)


def test_calibration_metrics_reward_honest_probabilities(rng):
    """A well-calibrated set must score better than a systematically shifted one."""
    y = rng.integers(0, 2, 2000)
    honest = np.where(y == 1, rng.beta(8, 2, 2000), rng.beta(2, 8, 2000))
    overconfident = np.clip(honest + 0.30, 0.0, 1.0)

    good = calibration_metrics(honest, y)
    bad = calibration_metrics(overconfident, y)
    assert good["brier_score"] < bad["brier_score"]
    assert good["expected_calibration_error"] < bad["expected_calibration_error"]


def test_calibration_ignores_samples_never_held_out(rng):
    probabilities = np.array([0.1, np.nan, 0.9, np.nan])
    y = np.array([0, 1, 1, 0])
    assert calibration_metrics(probabilities, y)["n_scored"] == 2
