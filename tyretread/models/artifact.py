"""Versioned model artifacts.

The original project fitted an SVM inside its evaluation script and then discarded
it, so nothing existed to deploy and the reported metrics described a model no user
could ever have been served (docs/AUDIT.md 3.10). An artifact here is deliberately
self-contained: it carries everything needed to reproduce an inference bit-for-bit,
and enough provenance to say which experiment produced it.

Feature order is stored explicitly. A scaler and a coefficient vector are
meaningless without knowing which column is which, and a silent column reordering
between training and serving is the kind of bug that produces plausible, wrong
answers rather than a crash.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from ..config import ARTIFACT_DIR

__all__ = ["ArtifactMetadata", "ModelArtifact", "save_artifact", "load_artifact",
           "list_artifacts"]

ARTIFACT_FORMAT_VERSION = 1


@dataclass
class ArtifactMetadata:
    """Everything about a model except the fitted estimator itself."""

    model_id: str
    model_name: str
    #: Canonical column order the estimator was fitted on.
    feature_names: list[str]
    #: 0/1 label meanings, e.g. {"0": "serviceable", "1": "worn"}.
    class_labels: dict[str, str]
    #: Probability of the positive class above which a tyre is called worn.
    decision_threshold: float
    #: Half-width of the band around the threshold in which the system abstains.
    abstain_band: float
    training_dataset: str
    n_training_samples: int
    validation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    quality_thresholds: dict[str, Any] = field(default_factory=dict)
    #: Training-set distribution of the reported features, used to place a new
    #: measurement in context. A percentile is meaningless without the population it
    #: is taken against, and that population must be the data the model actually saw.
    feature_reference: dict[str, Any] = field(default_factory=dict)
    #: Tread-versus-sidewall detector, stored as plain coefficients rather than a
    #: second binary. See tyretread.imaging.surface for why it reports rather than
    #: rejects. Empty for artifacts trained before it existed.
    surface_detector: dict[str, Any] = field(default_factory=dict)
    #: Features computed and reported but deliberately excluded from the model.
    #: legacy TSCI lives here: valid above the oversampling floor (exp004) but with no
    #: measurable predictive contribution (exp009).
    diagnostic_features: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    experiment_id: str | None = None
    notes: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    format_version: int = ARTIFACT_FORMAT_VERSION
    python_version: str = field(default_factory=platform.python_version)
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    numpy_version: str = field(default_factory=lambda: np.__version__)


@dataclass
class ModelArtifact:
    """A fitted estimator plus its metadata."""

    estimator: Any
    metadata: ArtifactMetadata

    def vectorise(self, features: dict[str, float]) -> np.ndarray:
        """Order a feature dict into the matrix row the estimator expects.

        Raises on a missing feature rather than substituting a default: a quietly
        zero-filled column is indistinguishable from a real measurement downstream.
        """
        missing = [n for n in self.metadata.feature_names if n not in features]
        if missing:
            raise ValueError(
                f"missing {len(missing)} feature(s) required by model "
                f"{self.metadata.model_id}: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        return np.array(
            [[float(features[n]) for n in self.metadata.feature_names]], dtype=np.float64
        )

    def probability_defect(self, features: dict[str, float]) -> float:
        """Calibrated probability that the tyre shows a visible defect."""
        row = self.vectorise(features)
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError(
                f"estimator {type(self.estimator).__name__} cannot produce "
                "probabilities; a model without calibrated probabilities cannot be "
                "used for an abstaining decision"
            )
        return float(self.estimator.predict_proba(row)[0, 1])


def save_artifact(
    artifact: ModelArtifact, directory: Path | None = None
) -> tuple[Path, Path]:
    """Persist ``artifact`` as a joblib estimator plus a readable JSON sidecar.

    The metadata is written as plain JSON rather than pickled with the estimator so
    that a human, or a CI job, can inspect what a deployed model claims without
    unpickling anything.
    """
    directory = directory or ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / f"{artifact.metadata.model_id}.joblib"
    meta_path = directory / f"{artifact.metadata.model_id}.json"

    joblib.dump(artifact.estimator, model_path, compress=3)
    meta_path.write_text(json.dumps(asdict(artifact.metadata), indent=2, default=str) + "\n")
    return model_path, meta_path


def load_artifact(model_id: str, directory: Path | None = None) -> ModelArtifact:
    """Load a persisted artifact, refusing anything it cannot vouch for."""
    directory = directory or ARTIFACT_DIR
    model_path = directory / f"{model_id}.joblib"
    meta_path = directory / f"{model_id}.json"

    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"artifact '{model_id}' is incomplete in {directory} "
            f"(estimator={model_path.is_file()}, metadata={meta_path.is_file()})"
        )

    raw = json.loads(meta_path.read_text())
    version = raw.get("format_version")
    if version != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"artifact '{model_id}' has format version {version}, "
            f"this build understands {ARTIFACT_FORMAT_VERSION}"
        )

    metadata = ArtifactMetadata(**raw)
    if not metadata.feature_names:
        raise ValueError(f"artifact '{model_id}' declares no feature names")

    return ModelArtifact(estimator=joblib.load(model_path), metadata=metadata)


def list_artifacts(directory: Path | None = None) -> list[str]:
    directory = directory or ARTIFACT_DIR
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
