"""Central configuration.

Every tunable that affects a measurement lives here, so that an experiment can
be described by this module's values plus a git commit. Paths are overridable by
environment variable so the same code runs locally, in CI and on a server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _env_path("TYRETREAD_DATA_DIR", PROJECT_ROOT / "data")
EXTERNAL_DATA_DIR = _env_path("TYRETREAD_EXTERNAL_DATA_DIR", DATA_DIR / "external")
ARTIFACT_DIR = _env_path("TYRETREAD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
OUTPUT_DIR = _env_path("TYRETREAD_OUTPUT_DIR", PROJECT_ROOT / "outputs")
EXPERIMENT_DIR = _env_path("TYRETREAD_EXPERIMENT_DIR", PROJECT_ROOT / "experiments")


@dataclass(frozen=True)
class ScaleConfig:
    """Resolution normalisation.

    The original pipeline resized every ROI to a fixed 256x128 regardless of the
    source resolution. Because upsampling cannot create detail while downsampling
    packs real detail into high spatial frequencies, that made every
    frequency-domain feature a function of the source image's resolution and
    compression history. Measured effect: +0.266 TSCI across a 128->1024 px
    source-width sweep on identical tyres, versus a 0.074 good-vs-bad signal
    (docs/AUDIT.md 3.2, experiments/exp001_resolution_confound).

    The fix has two halves:
      * never upsample - an image that lacks detail must be rejected, not invented
      * analyse every image at one common pixel scale
    """

    analysis_width: int = 256
    analysis_height: int = 128
    #: An ROI smaller than this in either axis cannot be analysed without
    #: upsampling, so it is refused by the quality gate instead.
    min_native_width: int = 256
    min_native_height: int = 128
    allow_upsampling: bool = False


@dataclass(frozen=True)
class PreprocessConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    gaussian_ksize: tuple[int, int] = (5, 5)
    #: Longest edge a submitted photograph is reduced to before any analysis.
    #: Phone cameras produce 4000+ px images; nothing in the pipeline benefits.
    max_input_edge: int = 1600


@dataclass(frozen=True)
class RoiConfig:
    centre_band_top: float = 0.20
    centre_band_bottom: float = 0.80
    canny_low: int = 50
    canny_high: int = 150
    close_kernel: tuple[int, int] = (15, 15)
    min_aspect_ratio: float = 2.0
    min_width_fraction: float = 0.40
    #: A contour must also fill a meaningful share of the centre band before it is
    #: preferred over it. Without these, a thin horizontal sliver satisfied the aspect
    #: and width tests and was accepted as "the tread": re-encoding one photograph at
    #: JPEG quality 95 moved the ROI from the full 1592x955 band to a 649x100 strip,
    #: dropping oversampling from 6.22 to 0.78 and flipping the image from assessable
    #: to refused. A contour that small is not a tread band.
    min_height_fraction: float = 0.35
    min_band_coverage: float = 0.25


@dataclass(frozen=True)
class SpectralConfig:
    #: Band over which the log-log power-spectrum slope is fitted, as a fraction
    #: of the Nyquist radius. The top of the band stops short of Nyquist because
    #: any resampling applies an anti-alias filter there.
    slope_band: tuple[float, float] = (0.08, 0.60)
    orientation_bins: int = 36
    #: Legacy TSCI threshold, retained only to reproduce the original result.
    legacy_hf_radius_fraction: float = 0.20


@dataclass(frozen=True)
class TextureConfig:
    #: 256 grey levels makes the GLCM extremely sparse at these ROI sizes and is
    #: slow; 32 levels is the usual compromise in the texture literature.
    levels: int = 32
    distances: tuple[int, ...] = (1, 2, 4)
    #: Averaging over four angles makes the descriptors rotation-insensitive,
    #: which matters because a phone is not held at a fixed angle to the tread.
    angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    lbp_radius: int = 2
    lbp_points: int = 16


@dataclass(frozen=True)
class Config:
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    random_seed: int = 42


CONFIG = Config()
