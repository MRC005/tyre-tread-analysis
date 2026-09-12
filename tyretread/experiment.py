"""Lightweight experiment tracking.

Every measurement quoted in the documentation should be traceable to a record
written by this module: what was run, on which data, with which configuration and
seed, what came out, and what was decided as a result. The alternative - numbers
remembered from a terminal session - is how the original project came to publish a
result whose headline feature turned out to be an artefact.

Records are plain JSON next to a human-readable Markdown log, so they diff cleanly
in Git and need no service to read.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import EXPERIMENT_DIR

__all__ = ["ExperimentRecord", "record_experiment", "rebuild_log"]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


@dataclass
class ExperimentRecord:
    """One experiment, fully described.

    ``interpretation`` and ``decision`` are mandatory in spirit: a metric with no
    stated consequence is how a project accumulates numbers nobody acts on. A
    negative result with a clear decision is a successful experiment.
    """

    experiment_id: str
    title: str
    hypothesis: str
    dataset: str
    method: str
    validation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    decision: str = ""
    seed: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    git_commit: str | None = field(default_factory=_git_commit)
    git_dirty: bool | None = field(default_factory=_dirty)
    python: str = field(default_factory=platform.python_version)


def record_experiment(record: ExperimentRecord, directory: Path | None = None) -> Path:
    """Write ``record`` as JSON, then regenerate the Markdown log.

    The JSON files are the source of truth and the log is derived from them, so
    re-running an experiment replaces its entry instead of appending a second copy.
    """
    directory = directory or EXPERIMENT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{record.experiment_id}.json"
    path.write_text(json.dumps(asdict(record), indent=2, default=str) + "\n")
    rebuild_log(directory)
    return path


def rebuild_log(directory: Path | None = None) -> Path:
    """Regenerate ``LOG.md`` from every experiment record in ``directory``."""
    directory = directory or EXPERIMENT_DIR
    records: list[dict[str, Any]] = []
    for json_path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(json_path.read_text()))
        except Exception:
            continue

    lines = [
        "# Experiment log",
        "",
        "Generated from the JSON records in this directory by",
        "`tyretread.experiment.rebuild_log` - do not edit by hand. Each entry states a",
        "hypothesis, what was run, what came out and what was decided as a result.",
        "A negative result with a clear decision is a successful experiment.",
        "",
    ]
    for rec in sorted(records, key=lambda r: r.get("experiment_id", "")):
        stamp = rec.get("timestamp", "")
        if rec.get("git_commit"):
            stamp += f" · commit `{rec['git_commit']}`"
        if rec.get("git_dirty"):
            stamp += " · working tree dirty"
        lines += [
            "---",
            "",
            f"## {rec.get('experiment_id')} — {rec.get('title')}",
            "",
            f"*{stamp}*",
            "",
            f"**Hypothesis.** {rec.get('hypothesis', '')}",
            "",
            f"**Data.** {rec.get('dataset', '')}",
            "",
            f"**Method.** {rec.get('method', '')}",
            "",
            f"**Validation.** {rec.get('validation', '')}",
            "",
            f"**Result.** {rec.get('interpretation', '')}",
            "",
            f"**Decision.** {rec.get('decision', '')}",
            "",
            f"Full record: [`{rec.get('experiment_id')}.json`]({rec.get('experiment_id')}.json)",
            "",
        ]

    log = directory / "LOG.md"
    log.write_text("\n".join(lines))
    return log
