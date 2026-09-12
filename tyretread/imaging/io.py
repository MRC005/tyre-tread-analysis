"""Image loading, EXIF orientation and input size reduction.

A photograph arriving from a phone browser differs from a file on disk in two ways
that break naive loading: it usually carries an EXIF orientation tag that must be
applied manually (OpenCV ignores EXIF), and it is often 4000 px or more on the
long edge, which costs time and memory for no analytical benefit.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import CONFIG, PreprocessConfig

__all__ = ["LoadedImage", "load_image", "decode_image_bytes"]


@dataclass(frozen=True)
class LoadedImage:
    """A decoded photograph plus the provenance needed to audit a decision."""

    bgr: np.ndarray
    native_width: int
    native_height: int
    exif_orientation: int | None
    downscaled: bool

    @property
    def width(self) -> int:
        return self.bgr.shape[1]

    @property
    def height(self) -> int:
        return self.bgr.shape[0]

    def as_dict(self) -> dict[str, object]:
        return {
            "native_width": self.native_width,
            "native_height": self.native_height,
            "analysis_width": self.width,
            "analysis_height": self.height,
            "exif_orientation": self.exif_orientation,
            "downscaled": self.downscaled,
        }


def _from_pil(pil: Image.Image, config: PreprocessConfig) -> LoadedImage:
    orientation = None
    try:
        exif = pil.getexif()
        orientation = exif.get(274) if exif else None
    except Exception:
        # A malformed or absent EXIF block is normal, not an error.
        orientation = None

    # Applies the orientation tag and strips it, so the array is upright.
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    native_w, native_h = pil.size

    downscaled = False
    long_edge = max(native_w, native_h)
    if long_edge > config.max_input_edge:
        scale = config.max_input_edge / long_edge
        pil = pil.resize(
            (max(1, round(native_w * scale)), max(1, round(native_h * scale))),
            Image.LANCZOS,
        )
        downscaled = True

    rgb = np.asarray(pil)
    return LoadedImage(
        bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        native_width=native_w,
        native_height=native_h,
        exif_orientation=orientation,
        downscaled=downscaled,
    )


def load_image(path: str | Path, config: PreprocessConfig | None = None) -> LoadedImage:
    """Load a photograph from disk, honouring EXIF orientation."""
    config = config or CONFIG.preprocess
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    try:
        with Image.open(path) as pil:
            return _from_pil(pil, config)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"could not decode image {path}: {exc}") from exc


def decode_image_bytes(
    payload: bytes, config: PreprocessConfig | None = None
) -> LoadedImage:
    """Decode an uploaded photograph from raw bytes.

    Used by the API so an upload never has to touch the filesystem.
    """
    config = config or CONFIG.preprocess
    if not payload:
        raise ValueError("empty image payload")
    try:
        with Image.open(io.BytesIO(payload)) as pil:
            return _from_pil(pil, config)
    except Exception as exc:
        raise ValueError(f"could not decode uploaded image: {exc}") from exc
