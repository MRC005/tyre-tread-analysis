"""API contract tests.

These cover the failure paths as carefully as the success path, because a screening
tool is judged on what it does with a bad photograph, a hostile upload or a missing
model - not on what it does when everything is fine.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tyretread.api.app import create_app
from tyretread.api.settings import ApiSettings
from tyretread.models.artifact import ArtifactMetadata, ModelArtifact, save_artifact


@pytest.fixture
def trained_artifact(tmp_path, rng, grooved_tread, worn_tread):
    """A real artifact fitted on features from the synthetic fixtures.

    Genuinely fitted rather than mocked, so the tests exercise the real vectorise ->
    predict_proba -> decide -> explain path. Its accuracy is irrelevant here; the
    contract is what is under test.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from tyretread.features.extract import extract_features, feature_names

    names = feature_names()
    rows, labels = [], []
    for source, label in ((grooved_tread, 0), (worn_tread, 1)):
        for _ in range(20):
            noisy = np.clip(
                source.astype(np.int16) + rng.normal(0, 4, source.shape).astype(np.int16),
                0, 255,
            ).astype(np.uint8)
            extraction = extract_features(noisy)
            assert extraction.usable
            rows.append([extraction.features[n] for n in names])
            labels.append(label)

    X = np.array(rows)
    y = np.array(labels)
    estimator = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        method="sigmoid", cv=3, ensemble=True,
    ).fit(X, y)

    from tyretread.models.evidence_features import REPORTED_FEATURES, build_reference
    reference = build_reference(
        {n: X[:, names.index(n)] for n in REPORTED_FEATURES if n in names}, y
    )

    artifact = ModelArtifact(
        estimator=estimator,
        metadata=ArtifactMetadata(
            model_id="test_current", model_name="logistic_l2", feature_names=names,
            class_labels={"0": "serviceable", "1": "worn"},
            decision_threshold=0.5, abstain_band=0.10,
            training_dataset="synthetic fixtures", n_training_samples=len(labels),
            validation="not validated - test fixture only",
            # Shaped like a real artifact's metrics record, including the bulky parts
            # the API is supposed to strip, so the summarising is actually exercised.
            metrics={
                "ranking": [{"model": "logistic_l2", "balanced_accuracy": 0.9}],
                "winner": {
                    "metrics": {
                        "balanced_accuracy": {"mean": 0.91, "std": 0.02, "n_folds": 25},
                        "roc_auc": {"mean": 0.97, "std": 0.01, "n_folds": 25},
                    },
                    "confusion_matrix": [[10, 1], [2, 9]],
                },
                "calibration": {"brier_score": 0.07, "expected_calibration_error": 0.07},
                "threshold_choice": {"abstention_rate": 0.17,
                                     "balanced_accuracy_on_decided": 0.96},
                "quality_gate_pass_rate": 0.92,
                "data_hygiene": {"n_exact_groups": 28},
            },
            feature_reference=reference,
            diagnostic_features=["legacy_tsci"],
        ),
    )
    save_artifact(artifact, tmp_path)
    return artifact, tmp_path


@pytest.fixture
def client(trained_artifact, monkeypatch):
    _, directory = trained_artifact
    monkeypatch.setenv("TYRETREAD_ARTIFACT_DIR", str(directory))
    # config caches ARTIFACT_DIR at import, so patch where load_artifact resolves it
    import tyretread.models.artifact as artifact_module
    monkeypatch.setattr(artifact_module, "ARTIFACT_DIR", directory)

    app = create_app(ApiSettings(model_id="test_current", max_upload_bytes=2 * 1024 * 1024))
    with TestClient(app) as test_client:
        yield test_client


def _jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buffer.tobytes()


def _post(client, bgr, **params):
    return client.post(
        "/v1/inspect",
        files={"image": ("tyre.jpg", io.BytesIO(_jpeg(bgr)), "image/jpeg")},
        params=params,
    )


# --------------------------------------------------------------------------
# Meta endpoints
# --------------------------------------------------------------------------

def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_id"] == "test_current"


def test_health_is_degraded_rather_than_failing_without_a_model(tmp_path, monkeypatch):
    import tyretread.models.artifact as artifact_module
    monkeypatch.setattr(artifact_module, "ARTIFACT_DIR", tmp_path)
    with TestClient(create_app(ApiSettings(model_id="absent"))) as client:
        response = client.get("/health")
        assert response.status_code == 200, "health must answer even when unusable"
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["detail"]


def test_inspecting_without_a_model_returns_503(tmp_path, monkeypatch, grooved_tread):
    import tyretread.models.artifact as artifact_module
    monkeypatch.setattr(artifact_module, "ARTIFACT_DIR", tmp_path)
    with TestClient(create_app(ApiSettings(model_id="absent"))) as client:
        response = _post(client, grooved_tread)
        assert response.status_code == 503
        assert response.json()["error"] == "model_unavailable"


def test_model_endpoint_states_how_the_model_was_validated(client):
    body = client.get("/v1/model").json()
    assert body["id"] == "test_current"
    assert body["validation"]
    assert body["n_training_samples"] > 0


# --------------------------------------------------------------------------
# The success path
# --------------------------------------------------------------------------

def test_a_good_photograph_produces_a_full_report(client, grooved_tread):
    response = _post(client, grooved_tread)
    assert response.status_code == 200
    body = response.json()

    assert body["result"]["verdict"] in {"likely_serviceable", "defect_suspected", "inconclusive"}
    assert body["image_quality"]["usable"] is True
    assert body["measurements"]
    assert body["explanation"]["exact"] is True
    assert body["explanation"]["reasons"]
    assert body["evidence"]["roi"].startswith("data:image/")
    assert body["disclaimer"]
    assert body["elapsed_ms"] > 0


def test_probability_and_confidence_are_both_reported(client, grooved_tread):
    result = _post(client, grooved_tread).json()["result"]
    assert 0.0 <= result["probability_defect"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_evidence_can_be_switched_off_for_slow_connections(client, grooved_tread):
    with_evidence = len(_post(client, grooved_tread).content)
    without = len(_post(client, grooved_tread, include_evidence=False).content)
    assert without < with_evidence / 2
    assert _post(client, grooved_tread, include_evidence=False).json()["evidence"] is None


def test_raw_features_are_opt_in(client, grooved_tread):
    assert _post(client, grooved_tread).json()["features"] is None
    features = _post(client, grooved_tread, include_features=True).json()["features"]
    assert len(features) > 40


def test_every_explanation_contribution_names_a_real_feature(client, grooved_tread):
    body = _post(client, grooved_tread, include_features=True).json()
    feature_keys = set(body["features"])
    for contribution in body["explanation"]["contributions"]:
        assert contribution["feature"] in feature_keys, (
            "an explanation must refer to measurements that were actually taken"
        )


# --------------------------------------------------------------------------
# Bad photographs are results, not errors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name,expected_issue", [
    ("blurred_image", "too_blurry"),
    ("dark_image", "too_dark"),
])
def test_unusable_photographs_return_200_with_advice(client, request, fixture_name, expected_issue):
    bgr = request.getfixturevalue(fixture_name)
    response = _post(client, bgr)
    assert response.status_code == 200, "a refusal is a result, not an HTTP error"
    body = response.json()
    assert body["result"]["verdict"] == "unable_to_assess"
    assert body["result"]["probability_defect"] is None
    assert expected_issue in body["image_quality"]["issues"]
    assert body["image_quality"]["advice"], "a refusal must tell the user what to do"


def test_a_tiny_image_is_refused_for_resolution(client, grooved_tread):
    tiny = cv2.resize(grooved_tread, (120, 80), interpolation=cv2.INTER_AREA)
    body = _post(client, tiny).json()
    assert body["result"]["verdict"] == "unable_to_assess"
    assert "resolution_too_low" in body["image_quality"]["issues"]


def test_a_refused_image_still_returns_visual_evidence(client, blurred_image):
    """Seeing what the system looked at is how a user understands the refusal."""
    evidence = _post(client, blurred_image).json()["evidence"]
    assert evidence["original"].startswith("data:image/")
    assert evidence["roi"].startswith("data:image/")


# --------------------------------------------------------------------------
# Hostile and malformed input
# --------------------------------------------------------------------------

def test_a_non_image_upload_is_rejected_as_undecodable(client):
    response = client.post(
        "/v1/inspect",
        files={"image": ("notes.txt", io.BytesIO(b"this is not an image"), "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "undecodable_image"


def test_an_unsupported_content_type_is_rejected(client):
    response = client.post(
        "/v1/inspect",
        files={"image": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_media_type"


def test_an_oversized_upload_is_rejected_before_decoding(client):
    oversized = io.BytesIO(b"\xff\xd8" + b"\x00" * (3 * 1024 * 1024))
    response = client.post(
        "/v1/inspect",
        files={"image": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "file_too_large"


def test_an_empty_upload_is_rejected(client):
    response = client.post(
        "/v1/inspect", files={"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert response.status_code in (400, 422)


def test_a_missing_file_field_is_a_validation_error(client):
    assert client.post("/v1/inspect").status_code == 422


def test_a_truncated_jpeg_does_not_crash_the_server(client, grooved_tread):
    truncated = _jpeg(grooved_tread)[: len(_jpeg(grooved_tread)) // 3]
    response = client.post(
        "/v1/inspect", files={"image": ("cut.jpg", io.BytesIO(truncated), "image/jpeg")},
    )
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert "result" in response.json()


# --------------------------------------------------------------------------
# Cross-cutting
# --------------------------------------------------------------------------

def test_every_response_carries_a_request_id(client, grooved_tread):
    assert _post(client, grooved_tread).headers["x-request-id"]


def test_a_supplied_request_id_is_echoed(client, grooved_tread):
    response = client.post(
        "/v1/inspect",
        files={"image": ("t.jpg", io.BytesIO(_jpeg(grooved_tread)), "image/jpeg")},
        headers={"x-request-id": "trace-me-123"},
    )
    assert response.headers["x-request-id"] == "trace-me-123"


def test_cors_headers_are_returned_for_the_configured_origin(client, grooved_tread):
    response = client.post(
        "/v1/inspect",
        files={"image": ("t.jpg", io.BytesIO(_jpeg(grooved_tread)), "image/jpeg")},
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_the_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/inspect" in schema["paths"]


# --------------------------------------------------------------------------
# Measured evidence versus model interpretation
# --------------------------------------------------------------------------

def test_measured_evidence_is_returned_with_percentiles(client, grooved_tread):
    body = _post(client, grooved_tread).json()
    evidence = body["evidence_features"]
    assert evidence["measurements"]
    for measurement in evidence["measurements"]:
        assert 0.0 <= measurement["percentile"] <= 100.0
        assert measurement["band"] in {"very low", "low", "typical", "high", "very high"}
        assert measurement["description"]


def test_diagnostics_are_reported_but_are_not_model_inputs(client, grooved_tread):
    """exp009: TSCI is computed and shown, but excluded from the classifier."""
    body = _post(client, grooved_tread, include_features=True).json()
    diagnostics = body["evidence_features"]["diagnostics"]
    assert "legacy_tsci" in diagnostics

    # The authoritative check: the artifact's feature list must exclude it.
    from tyretread.models.artifact import load_artifact
    import tyretread.models.artifact as artifact_module
    artifact = load_artifact("test_current", artifact_module.ARTIFACT_DIR)
    assert "legacy_tsci" not in artifact.metadata.feature_names


def test_evidence_and_explanation_are_separate_fields(client, grooved_tread):
    """The contract that keeps measurement distinct from interpretation."""
    body = _post(client, grooved_tread).json()
    assert "evidence_features" in body
    assert "explanation" in body
    assert body["evidence_features"]["note"]


def test_an_agreement_statement_accompanies_the_evidence(client, grooved_tread):
    body = _post(client, grooved_tread).json()
    agreement = body["evidence_features"]["agreement"]
    assert agreement is None or isinstance(agreement, str)


def test_the_disclaimer_disclaims_tread_depth(client, grooved_tread):
    """The product was reframed to condition screening; the disclaimer must say so."""
    disclaimer = _post(client, grooved_tread).json()["disclaimer"].lower()
    assert "cannot measure tread depth" in disclaimer
    assert "not a certified inspection" in disclaimer


def test_no_response_field_promises_a_depth_measurement(client, grooved_tread):
    """No response may state a tread depth.

    Matches the *claim* rather than the words. An earlier version forbade the substring
    "tread depth is", which flagged the response's own disclaimer ("does not tell you
    how much tread depth is left") - the copy was correct and the guard was wrong. What
    must never appear is a numeric depth, in any unit.
    """
    import json as _json
    import re as _re

    payload = _post(client, grooved_tread).json()
    assert "probability_worn" not in _json.dumps(payload), (
        "renamed to probability_defect when reframed"
    )

    # Scan the prose only. Base64 evidence blobs are arbitrary alphanumerics and will
    # contain sequences like "0mm" by chance - an earlier version of this test searched
    # the whole serialised response and failed on image data, which says nothing about
    # what the product claims.
    def prose(node: object) -> list[str]:
        if isinstance(node, str):
            return [] if node.startswith("data:") else [node]
        if isinstance(node, dict):
            return [t for v in node.values() for t in prose(v)]
        if isinstance(node, list):
            return [t for v in node for t in prose(v)]
        return []

    text = " ".join(prose(payload)).lower()

    numeric_depth = _re.search(r"\d+(\.\d+)?\s*(mm\b|millimet)", text)
    assert numeric_depth is None, f"response states a depth: {numeric_depth.group(0)!r}"

    for claim in ("tread depth of", "depth is approximately", "measured depth"):
        assert claim not in text


# --------------------------------------------------------------------------
# Response size and contract stability
# --------------------------------------------------------------------------

def test_model_metrics_are_a_compact_summary_not_the_experiment_record(client, grooved_tread):
    """The artifact's full metrics record is 3.3 KB of internal experiment detail.

    Serialising it onto every inspection made a phone on mobile data pay for a model
    ranking, confusion matrices and data-hygiene counts it has no use for, and coupled
    the client to the shape of an internal record.
    """
    import json

    metrics = _post(client, grooved_tread).json()["model"]["metrics"]
    assert metrics is not None
    # Only the summary fields, nothing nested.
    assert set(metrics).issubset({
        "balanced_accuracy", "balanced_accuracy_std", "roc_auc", "brier_score",
        "expected_calibration_error", "abstention_rate",
        "balanced_accuracy_on_decided", "quality_gate_pass_rate",
    })
    for forbidden in ("ranking", "winner", "confusion_matrix", "data_hygiene"):
        assert forbidden not in metrics
    assert len(json.dumps(metrics)) < 600


def test_the_model_endpoint_serves_the_same_summary(client):
    metrics = client.get("/v1/model").json()["metrics"]
    assert metrics is not None
    assert "ranking" not in metrics


def test_a_data_saver_response_stays_small(client, grooved_tread):
    """include_evidence=false must actually remove the expensive part."""
    full = len(_post(client, grooved_tench := grooved_tread).content)
    lean = len(_post(client, grooved_tench, include_evidence=False).content)
    assert lean < 20_000, "a response without evidence should be a few KB"
    assert lean < full / 3
