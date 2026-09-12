"""Feature tests.

Two kinds appear here. Some check that a descriptor measures what its name claims,
using synthetic textures whose correct answer is known by construction. Others are
property tests asserting scale invariance - they encode the conclusion of exp002 as
an executable claim, so a future change to preprocessing that reintroduces a
resolution dependence fails the suite instead of quietly degrading the science.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tyretread.features.extract import extract_features, feature_names
from tyretread.features.spectral import (
    legacy_tsci, orientation_statistics, power_spectrum, spectral_slope,
)
from tyretread.features.texture import glcm_features, lbp_histogram, texture_features
from tyretread.imaging.preprocess import normalise_scale, preprocess
from tyretread.imaging.roi import extract_roi


def _normalised(bgr: np.ndarray) -> np.ndarray:
    return normalise_scale(extract_roi(preprocess(bgr).smoothed).roi)


# --------------------------------------------------------------------------
# Descriptors measure what they claim
# --------------------------------------------------------------------------

def test_directional_grooves_are_more_anisotropic_than_flat_noise(grooved_tread, worn_tread):
    grooved = orientation_statistics(power_spectrum(_normalised(grooved_tread)))
    flat = orientation_statistics(power_spectrum(_normalised(worn_tread)))
    assert grooved["orientation_anisotropy"] > flat["orientation_anisotropy"]
    assert grooved["orientation_coherence"] > flat["orientation_coherence"]


def test_flat_noise_spreads_energy_over_more_directions(grooved_tread, worn_tread):
    grooved = orientation_statistics(power_spectrum(_normalised(grooved_tread)))
    flat = orientation_statistics(power_spectrum(_normalised(worn_tread)))
    assert flat["orientation_entropy"] > grooved["orientation_entropy"]


def test_white_noise_has_a_flatter_spectrum_than_smooth_gradient(rng):
    noise = rng.integers(0, 255, (256, 256), dtype=np.uint8)
    smooth = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    noise_slope, _ = spectral_slope(power_spectrum(noise))
    smooth_slope, _ = spectral_slope(power_spectrum(smooth))
    assert noise_slope < smooth_slope


def test_glcm_is_computed_at_every_configured_distance(grooved_tread):
    values = glcm_features(_normalised(grooved_tread))
    for distance in (1, 2, 4):
        assert f"glcm_contrast_d{distance}" in values
    assert "glcm_contrast_anisotropy" in values


def test_lbp_histogram_is_normalised(grooved_tread):
    hist = lbp_histogram(_normalised(grooved_tread))
    assert sum(hist.values()) == pytest.approx(1.0, abs=1e-6)
    assert len(hist) == 18  # lbp_points=16 -> P + 2 uniform bins


def test_lbp_is_actually_included_in_the_feature_vector():
    """The original pipeline computed LBP then discarded it (audit 3.10)."""
    names = feature_names()
    assert any(n.startswith("lbp_") for n in names)


def test_every_feature_is_finite_on_a_real_photograph(sample_photograph):
    extraction = extract_features(sample_photograph)
    assert extraction.usable
    bad = {k: v for k, v in extraction.features.items() if not np.isfinite(v)}
    assert not bad, f"non-finite features: {bad}"


def test_feature_names_match_what_extraction_produces(sample_photograph):
    extraction = extract_features(sample_photograph)
    assert sorted(extraction.features) == sorted(feature_names())


# --------------------------------------------------------------------------
# Scale invariance: exp002 and exp004's conclusions, as executable properties
# --------------------------------------------------------------------------
#
# These use real photographs, not the synthetic fixtures. Synthetic textures are
# excellent for checking that a descriptor measures what it claims, but their
# spectral statistics are not those of real images, so they are the wrong substrate
# for a scale-invariance claim. Measured on real legacy photographs at 1.5x
# resolution change with both versions above the 3x oversampling floor, the median
# relative drift is 1.1% for spectral_slope and 0.3% for orientation_entropy; the
# 90th percentile is 4.8% and 3.7%. Tolerances below come from that measurement.
#
# The angular statistics are excluded from the tight tolerance on purpose: their
# median drift is under 2% but the tail reaches 178%, because circular statistics
# become ill-conditioned when a surface is nearly isotropic and the vector sum
# approaches zero. That is a real property of the descriptor, documented rather than
# hidden, and it is why the model receives the anisotropy value alongside the
# coherence rather than either alone.

#: p90 relative drift measured on real photographs, with headroom.
SCALE_TOLERANCE = {
    "spectral_slope": 0.10,
    "orientation_entropy": 0.08,
}

#: A 1.5x reduction must leave the smaller version above the 3x floor, so the source
#: needs at least 4.5x. Requiring 4.8x gives a little headroom.
MIN_SOURCE_OVERSAMPLING = 4.8


def _high_resolution_source() -> np.ndarray:
    """A real photograph with enough resolution to test invariance honestly.

    Skips rather than fails when the dataset is not present. The image datasets are
    not redistributable (docs/AUDIT.md 3.7) and are therefore not tracked in Git, so
    a checkout without them must skip these tests with a clear reason instead of
    appearing to pass.
    """
    import glob

    from tyretread.imaging.preprocess import oversampling_factor

    for path in sorted(glob.glob("data/good/*")) + sorted(glob.glob("data/bad/*")):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        roi = extract_roi(preprocess(bgr).smoothed).roi
        if oversampling_factor(roi.shape) >= MIN_SOURCE_OVERSAMPLING:
            return bgr

    pytest.skip(
        "no local image reaches "
        f"{MIN_SOURCE_OVERSAMPLING}x oversampling; scale-invariance tests need a "
        "high-resolution photograph. See docs/DATA.md for how to obtain the datasets."
    )


def _real_pair(reduction: float = 1.5):
    """The same real photograph at two resolutions, both above the 3x floor."""
    from tyretread.imaging.preprocess import oversampling_factor

    bgr = _high_resolution_source()
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (int(w / reduction), int(h / reduction)),
                       interpolation=cv2.INTER_AREA)

    large_roi = extract_roi(preprocess(bgr).smoothed).roi
    small_roi = extract_roi(preprocess(small).smoothed).roi
    factors = (oversampling_factor(large_roi.shape), oversampling_factor(small_roi.shape))
    return normalise_scale(large_roi), normalise_scale(small_roi), factors


@pytest.mark.parametrize("descriptor,tolerance", SCALE_TOLERANCE.items())
def test_descriptors_survive_a_resolution_change(descriptor, tolerance):
    """The same tread at two source resolutions must describe the same way.

    This is the property the original pipeline lacked below its oversampling floor:
    exp001 measured TSCI moving by 79-118% on identical tyres there.
    """
    from tyretread.features.spectral import spectral_features

    large, small, factors = _real_pair()
    a = spectral_features(large).as_dict()[descriptor]
    b = spectral_features(small).as_dict()[descriptor]
    assert abs(a - b) <= tolerance * abs(a), (
        f"{descriptor} moved {a:.4f} -> {b:.4f} on the same tread at "
        f"1.5x lower resolution (oversampling {factors[0]:.1f}x and {factors[1]:.1f}x)"
    )


def test_the_invariance_claim_is_only_made_above_the_floor():
    """Both versions in the invariance test must clear the 3x oversampling floor.

    Guards the test itself: if a future change to the ROI or the analysis grid pushed
    these below the floor, the invariance assertions would be testing behaviour
    outside the regime in which they were validated, and would silently mean nothing.
    """
    _, _, factors = _real_pair()
    assert min(factors) >= 3.0, (
        f"invariance fixtures dropped to {min(factors):.2f}x oversampling; "
        "re-derive the tolerances before trusting the tests above"
    )


def test_legacy_tsci_degenerates_below_the_oversampling_floor():
    """Documents exp004: TSCI's flaw is a missing precondition, not the formula.

    Below the floor, the same tread yields very different TSCI values. If a future
    change ever makes TSCI stable here, exp001 and exp004 should be revisited.
    """
    bgr = _high_resolution_source()
    h, w = bgr.shape[:2]
    tiny = cv2.resize(bgr, (256, int(h * 256 / w)), interpolation=cv2.INTER_AREA)

    full = legacy_tsci(extract_roi(preprocess(bgr).smoothed).roi)
    starved = legacy_tsci(extract_roi(preprocess(tiny).smoothed).roi)
    assert abs(full - starved) > 0.05 * abs(full), (
        f"legacy TSCI was expected to degenerate below the oversampling floor but "
        f"moved only {full:.4f} -> {starved:.4f}; revisit exp001 and exp004"
    )


def test_spectral_slope_reports_its_own_unreliability():
    """``spectral_slope_r2`` must flag content the power-law model cannot describe.

    The slope is only interpretable when a single power law fits. On real tread the
    fit is good (r-squared 0.93 mean across the legacy dataset), but on narrowband
    content it is meaningless - and the pipeline says so rather than reporting a
    confident number, which is what lets the explanation layer suppress it.
    """
    from tyretread.features.spectral import spectral_features

    narrowband = np.tile(
        (128 + 120 * np.sin(2 * np.pi * np.arange(256) / 16.0)).astype(np.uint8), (128, 1)
    )
    assert spectral_features(narrowband).as_dict()["spectral_slope_r2"] < 0.5

    real = normalise_scale(extract_roi(preprocess(_high_resolution_source()).smoothed).roi)
    assert spectral_features(real).as_dict()["spectral_slope_r2"] > 0.80


# --------------------------------------------------------------------------
# ROI stability
# --------------------------------------------------------------------------

def test_roi_selection_survives_a_jpeg_re_encode(sample_photograph):
    """Regression: a thin sliver used to be accepted as the tread band.

    The contour rule tested only aspect ratio and width, so a 649x100 strip satisfied
    it. Re-encoding one photograph at JPEG quality 95 moved the ROI from the full
    1592x955 centre band to that strip, dropping oversampling from 6.22 to 0.78 and
    flipping the image from assessable to refused. Trivial re-encoding must not change
    what the system looks at.
    """
    from tyretread.imaging.io import decode_image_bytes
    from tyretread.imaging.preprocess import oversampling_factor

    baseline = extract_roi(preprocess(sample_photograph).smoothed)
    baseline_factor = oversampling_factor(baseline.roi.shape)

    for quality in (95, 90, 85, 75):
        ok, buffer = cv2.imencode(".jpg", sample_photograph, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        decoded = decode_image_bytes(buffer.tobytes())
        roi = extract_roi(preprocess(decoded.bgr).smoothed)
        factor = oversampling_factor(roi.roi.shape)
        assert factor == pytest.approx(baseline_factor, rel=0.25), (
            f"oversampling moved {baseline_factor:.2f} -> {factor:.2f} at JPEG quality "
            f"{quality}; ROI selection has become unstable again"
        )


def test_a_thin_sliver_is_never_accepted_as_a_tread_band():
    """The specific shape that caused the instability."""
    from tyretread.config import CONFIG

    band_height = 955
    sliver_height = 100
    assert sliver_height < band_height * CONFIG.roi.min_height_fraction, (
        "the height floor must exclude a sliver of this proportion"
    )
