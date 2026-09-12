"""Regression tests for the correction at the centre of this revision.

exp001 showed that resizing every ROI to a fixed grid made TSCI a measurement of
source resolution: the artefact was 3.6x larger than the good-versus-worn signal.
The fix is that the pipeline must never upsample. These tests exist so that fix
cannot be undone by accident - they are the guard on a scientific result, not a
check on plumbing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tyretread.config import CONFIG, ScaleConfig
from tyretread.imaging.preprocess import normalise_scale, oversampling_factor


def test_refuses_to_upsample_a_small_roi():
    small = np.zeros((64, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="fabricate detail"):
        normalise_scale(small)


def test_refuses_when_only_one_axis_is_too_small():
    # 300 px wide is enough, 100 px tall is not. Both axes must clear the grid.
    with pytest.raises(ValueError):
        normalise_scale(np.zeros((100, 300), dtype=np.uint8))


def test_downsamples_a_large_roi_to_the_analysis_grid():
    out = normalise_scale(np.zeros((800, 1600), dtype=np.uint8))
    assert out.shape == (CONFIG.scale.analysis_height, CONFIG.scale.analysis_width)


def test_upsampling_only_happens_when_explicitly_allowed():
    config = dataclasses.replace(CONFIG.scale, allow_upsampling=True)
    out = normalise_scale(np.zeros((64, 100), dtype=np.uint8), config)
    assert out.shape == (config.analysis_height, config.analysis_width)


def test_oversampling_factor_is_the_binding_axis():
    # 1024/256 = 4.0 on width, 256/128 = 2.0 on height -> the smaller one wins.
    assert oversampling_factor((256, 1024)) == pytest.approx(2.0)


def test_oversampling_factor_below_one_means_unusable():
    assert oversampling_factor((64, 128)) < 1.0
