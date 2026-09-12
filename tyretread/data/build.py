"""Batch feature extraction over a dataset.

Records one row per image including images the quality gate refused, because the
refusal rate and its reasons are themselves a result: a gate that rejects most of a
dataset is telling you something about the data, not just about the images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

from ..config import Config, CONFIG
from ..features.extract import extract_features
from ..imaging.io import load_image
from ..imaging.quality import QualityThresholds
from .datasets import DatasetSpec, list_images

__all__ = ["build_feature_table"]


def build_feature_table(
    spec: DatasetSpec,
    *,
    config: Config | None = None,
    thresholds: QualityThresholds | None = None,
    include_legacy_tsci: bool = True,
    limit: int | None = None,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Extract features for every image in ``spec``.

    ``native_width``/``native_height``/``file_bytes`` are carried through
    deliberately. They are not model inputs - they are the confound the audit
    found (docs/AUDIT.md 3.3), and keeping them in the table is what allows a later
    experiment to prove the corrected features no longer depend on them.
    """
    config = config or CONFIG
    records = list_images(spec)
    if limit:
        records = records[:limit]

    rows: list[dict[str, object]] = []
    for i, record in enumerate(records, 1):
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(records)}", flush=True)

        row: dict[str, object] = {
            "dataset": record.dataset,
            "path": str(record.path),
            "filename": record.path.name,
            "source_dir": record.source_dir,
            "label": record.label,
        }
        try:
            loaded = load_image(record.path, config.preprocess)
        except Exception as exc:
            row.update({"usable": False, "error": f"load_failed: {exc}"})
            rows.append(row)
            continue

        row.update({
            "native_width": loaded.native_width,
            "native_height": loaded.native_height,
            "file_bytes": record.path.stat().st_size,
            "exif_orientation": loaded.exif_orientation,
        })

        try:
            extraction = extract_features(
                loaded.bgr, config=config, thresholds=thresholds,
                include_legacy_tsci=include_legacy_tsci,
            )
        except Exception as exc:
            row.update({"usable": False, "error": f"extract_failed: {exc}"})
            rows.append(row)
            continue

        row.update({
            "usable": extraction.usable,
            "error": None,
            "oversampling": extraction.oversampling,
            "roi_method": extraction.roi.method,
            "roi_coverage": extraction.roi.coverage,
            "quality_summary": extraction.quality.summary,
            "quality_issues": ";".join(i.value for i in extraction.quality.issues),
            "quality_warnings": ";".join(w.value for w in extraction.quality.warnings),
        })
        row.update(extraction.quality.metrics)
        row.update(extraction.features)
        rows.append(row)

    return pd.DataFrame(rows)
