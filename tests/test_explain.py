"""Explanations must be derived from the decision, not written to justify it."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tyretread.models.explain import FEATURE_DESCRIPTIONS, explain_prediction

NAMES = ["edge_density", "spectral_slope", "orientation_coherence", "glcm_homogeneity_d1"]


@pytest.fixture
def linear_model(rng):
    X = rng.normal(size=(300, 4))
    # Only the first and third features carry signal, with opposite signs.
    y = (2.0 * X[:, 0] - 1.5 * X[:, 2] > 0).astype(int)
    return CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
        method="sigmoid", cv=5, ensemble=True,
    ).fit(X, y), X, y


def test_a_linear_model_yields_an_exact_decomposition(linear_model):
    model, X, _ = linear_model
    explanation = explain_prediction(model, NAMES, dict(zip(NAMES, X[0])))
    assert explanation.exact is True
    assert len(explanation.contributions) == len(NAMES)


def test_the_informative_features_dominate_the_contributions(linear_model):
    """The explanation must recover the structure actually in the model."""
    model, X, _ = linear_model
    totals = {name: 0.0 for name in NAMES}
    for row in X[:80]:
        for contribution in explain_prediction(model, NAMES, dict(zip(NAMES, row))).contributions:
            totals[contribution.feature] += abs(contribution.contribution)

    ranked = sorted(totals, key=lambda n: -totals[n])
    assert set(ranked[:2]) == {"edge_density", "orientation_coherence"}, (
        f"explanation attributed the decision to {ranked[:2]}, but the model was "
        "built to depend on edge_density and orientation_coherence"
    )


def test_contribution_signs_follow_the_model_not_the_verdict(linear_model):
    """Feature 0 pushes towards worn, feature 2 away from it, by construction."""
    model, _, _ = linear_model
    high_first = explain_prediction(model, NAMES, {
        "edge_density": 3.0, "spectral_slope": 0.0,
        "orientation_coherence": 0.0, "glcm_homogeneity_d1": 0.0,
    })
    by_name = {c.feature: c for c in high_first.contributions}
    assert by_name["edge_density"].contribution > 0
    assert by_name["edge_density"].direction == "towards wear"

    high_third = explain_prediction(model, NAMES, {
        "edge_density": 0.0, "spectral_slope": 0.0,
        "orientation_coherence": 3.0, "glcm_homogeneity_d1": 0.0,
    })
    assert {c.feature: c for c in high_third.contributions}["orientation_coherence"].contribution < 0


def test_contributions_are_ordered_by_magnitude(linear_model):
    model, X, _ = linear_model
    contributions = explain_prediction(model, NAMES, dict(zip(NAMES, X[3]))).contributions
    magnitudes = [abs(c.contribution) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_reasons_are_plain_sentences_without_feature_names(linear_model):
    model, X, _ = linear_model
    for reason in explain_prediction(model, NAMES, dict(zip(NAMES, X[0]))).reasons:
        assert reason.endswith(".")
        assert reason[0].isupper()
        assert "glcm" not in reason.lower()
        assert "_" not in reason


def test_reasons_do_not_repeat_the_same_idea(linear_model):
    """Several LBP bins are one idea and must not become four near-identical lines."""
    model, X, _ = linear_model
    names = NAMES + [f"lbp_{i}" for i in range(6)]
    rng = np.random.default_rng(1)
    padded = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
        method="sigmoid", cv=5, ensemble=True,
    )
    X_padded = rng.normal(size=(300, len(names)))
    padded.fit(X_padded, (X_padded[:, 0] > 0).astype(int))

    reasons = explain_prediction(padded, names, dict(zip(names, X_padded[0]))).reasons
    assert len(reasons) == len(set(reasons))


def test_a_non_linear_model_refuses_to_invent_an_explanation(rng):
    """Honesty guard: no surrogate explanation is substituted."""
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    forest = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=30, random_state=0),
        method="sigmoid", cv=3, ensemble=True,
    ).fit(X, y)

    explanation = explain_prediction(forest, NAMES, dict(zip(NAMES, X[0])))
    assert explanation.exact is False
    assert explanation.reasons == [], "a non-linear model must not emit stated reasons"
    assert "cannot be decomposed" in explanation.method


def test_every_feature_family_has_a_plain_language_description():
    from tyretread.features.extract import feature_names
    from tyretread.models.explain import describe_feature

    for name in feature_names():
        description = describe_feature(name)
        assert description != name, f"{name} has no plain-language description"
        assert " " in description
