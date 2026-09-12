"""Model artifacts must round-trip, and must refuse what they cannot vouch for."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tyretread.models.artifact import (
    ARTIFACT_FORMAT_VERSION, ArtifactMetadata, ModelArtifact, load_artifact,
    save_artifact,
)

FEATURES = ["alpha", "beta", "gamma"]


@pytest.fixture
def artifact(rng):
    X = rng.normal(size=(150, 3))
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    estimator = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        method="sigmoid", cv=5, ensemble=True,
    ).fit(X, y)

    metadata = ArtifactMetadata(
        model_id="test_model_v1", model_name="logistic_l2", feature_names=FEATURES,
        class_labels={"0": "serviceable", "1": "worn"},
        decision_threshold=0.5, abstain_band=0.1,
        training_dataset="synthetic", n_training_samples=150,
        validation="5-fold x 2 repeats",
    )
    return ModelArtifact(estimator=estimator, metadata=metadata)


def test_round_trips_through_disk(artifact, tmp_path):
    save_artifact(artifact, tmp_path)
    loaded = load_artifact("test_model_v1", tmp_path)

    assert loaded.metadata.feature_names == FEATURES
    assert loaded.metadata.decision_threshold == artifact.metadata.decision_threshold

    row = {"alpha": 0.4, "beta": -0.2, "gamma": 1.1}
    assert loaded.probability_defect(row) == pytest.approx(artifact.probability_defect(row))


def test_metadata_is_readable_without_unpickling(artifact, tmp_path):
    _, meta_path = save_artifact(artifact, tmp_path)
    raw = json.loads(meta_path.read_text())
    assert raw["feature_names"] == FEATURES
    assert raw["format_version"] == ARTIFACT_FORMAT_VERSION
    assert raw["sklearn_version"]


def test_refuses_a_feature_dict_that_is_missing_a_column(artifact):
    with pytest.raises(ValueError, match="missing"):
        artifact.probability_defect({"alpha": 1.0, "beta": 2.0})


def test_feature_order_is_respected_not_dict_order(artifact):
    """A reordered dict must give an identical answer.

    Guards the failure mode a stored feature order exists to prevent: silently
    shuffled columns produce plausible, wrong predictions rather than a crash.
    """
    forward = {"alpha": 0.4, "beta": -0.2, "gamma": 1.1}
    reversed_dict = {"gamma": 1.1, "beta": -0.2, "alpha": 0.4}
    assert artifact.probability_defect(forward) == pytest.approx(
        artifact.probability_defect(reversed_dict)
    )


def test_vectorise_uses_the_declared_order(artifact):
    row = artifact.vectorise({"alpha": 1.0, "beta": 2.0, "gamma": 3.0})
    assert row.tolist() == [[1.0, 2.0, 3.0]]


def test_missing_files_raise_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_artifact("nope", tmp_path)


def test_an_unknown_format_version_is_refused(artifact, tmp_path):
    _, meta_path = save_artifact(artifact, tmp_path)
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = 999
    meta_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="format version"):
        load_artifact("test_model_v1", tmp_path)


def test_an_artifact_without_probabilities_is_refused(rng, tmp_path):
    from sklearn.svm import SVC

    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    bare = ModelArtifact(
        estimator=SVC(kernel="linear").fit(X, y),
        metadata=ArtifactMetadata(
            model_id="no_proba", model_name="svc", feature_names=FEATURES,
            class_labels={"0": "serviceable", "1": "worn"},
            decision_threshold=0.5, abstain_band=0.1,
            training_dataset="synthetic", n_training_samples=60, validation="none",
        ),
    )
    with pytest.raises(TypeError, match="calibrated probabilities"):
        bare.probability_defect({"alpha": 1.0, "beta": 0.0, "gamma": 0.0})
