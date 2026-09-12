from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260911)


def _multiscale_texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Broadband 1/f-like texture, as real rubber surfaces produce.

    Real tread photographs have a power-law power spectrum (measured r-squared 0.93
    on average across the legacy dataset). A single sine wave does not: its spectrum
    is a spike on a flat noise floor, which makes the power-law slope fit
    ill-conditioned and its value meaningless. Using a realistic broadband texture
    keeps the fixtures representative of what the pipeline actually sees.
    """
    field = np.zeros((h, w))
    amplitude = 1.0
    for octave in range(6):
        step = 2**octave
        coarse = rng.normal(0, amplitude, (h // step + 2, w // step + 2))
        field += cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        amplitude *= 0.6
    return field / (np.abs(field).max() + 1e-9)


@pytest.fixture
def grooved_tread(rng: np.random.Generator) -> np.ndarray:
    """A synthetic 'unworn' tread: deep directional grooves over rubber texture.

    Synthetic rather than a real photograph so the expected answer is known by
    construction. A test asserting that a real image scores a particular way only
    shows the pipeline has not changed, not that it measures the right thing.
    """
    h, w = 600, 900
    _, x = np.mgrid[0:h, 0:w]
    # Several groove harmonics, so the spectrum is grooved *and* broadband.
    grooves = (
        45 * np.sin(2 * np.pi * x / 26.0)
        + 18 * np.sin(2 * np.pi * x / 13.0)
        + 9 * np.sin(2 * np.pi * x / 7.0)
    )
    img = 128 + grooves + 40 * _multiscale_texture(rng, h, w)
    return np.stack([np.clip(img, 0, 255).astype(np.uint8)] * 3, axis=-1)


@pytest.fixture
def worn_tread(rng: np.random.Generator) -> np.ndarray:
    """A synthetic 'worn' tread: the same rubber texture, grooves flattened away.

    The texture amplitude is scaled so the fixture spans a realistic tonal range.
    An earlier version used a smaller amplitude and produced a dynamic range of 33,
    which the quality gate correctly refused as indistinguishable from an
    underexposed photograph - a real worn tyre photographed properly still spans
    most of the range. The fixture was wrong, not the gate.
    """
    h, w = 600, 900
    field = _multiscale_texture(rng, h, w)
    # Stretch to a realistic spread rather than a narrow band around mid-grey.
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    img = 40 + 175 * field
    return np.stack([np.clip(img, 0, 255).astype(np.uint8)] * 3, axis=-1)


@pytest.fixture
def blurred_image(grooved_tread: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(grooved_tread, (31, 31), 12)


@pytest.fixture
def dark_image(grooved_tread: np.ndarray) -> np.ndarray:
    return (grooved_tread * 0.06).astype(np.uint8)


@pytest.fixture
def bright_image(grooved_tread: np.ndarray) -> np.ndarray:
    return np.clip(grooved_tread.astype(np.int16) + 150, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_photograph() -> np.ndarray:
    """A real tyre photograph from the local dataset, or skip.

    The image datasets are not redistributable and so are not tracked in Git
    (docs/AUDIT.md 3.7). Tests that genuinely need a real photograph skip with a
    clear reason on a checkout without them, rather than failing or - worse -
    appearing to pass against a synthetic substitute.
    """
    import glob
    from pathlib import Path

    for candidate in ["data/images/test.jpg", *sorted(glob.glob("data/good/*"))]:
        if Path(candidate).is_file():
            from tyretread.imaging.io import load_image
            return load_image(candidate).bgr

    pytest.skip("no local tyre photograph available; see docs/DATA.md")


@pytest.fixture
def noisy_image(grooved_tread: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy sensor noise: preserves mean, contrast and sharpness, destroys texture."""
    noise = rng.normal(0, 30, grooved_tread.shape).astype(np.int16)
    return np.clip(grooved_tread.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def over_compressed_image(grooved_tread: np.ndarray) -> np.ndarray:
    """Quality-12 JPEG: the blockiness a messaging app leaves behind."""
    ok, buffer = cv2.imencode(".jpg", grooved_tread, [cv2.IMWRITE_JPEG_QUALITY, 12])
    assert ok
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


@pytest.fixture
def normal_phone_jpeg(grooved_tread: np.ndarray) -> np.ndarray:
    """Quality-85 JPEG: an ordinary phone photograph. Must NOT be refused."""
    ok, buffer = cv2.imencode(".jpg", grooved_tread, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


@pytest.fixture
def dark_but_well_exposed(grooved_tread: np.ndarray) -> np.ndarray:
    """A genuinely dark tyre, correctly exposed: low mean, but full dynamic range."""
    img = grooved_tread.astype(np.float32)
    # Compress towards the dark end while keeping a wide spread.
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return np.clip(img * 190.0 + 5.0, 0, 255).astype(np.uint8)
