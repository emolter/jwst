"""Tests for jwst.wfss_contam.trace_lut."""

import numpy as np
import pytest
from astropy.modeling.mappings import Mapping
from numpy.testing import assert_allclose

from jwst.wfss_contam.trace_lut import TraceLUT, _native_wavelength_grid, build_trace_lut

_ORDER = 1
_WMIN, _WMAX = 1.708, 2.28
_NAXIS = (2048, 2048)


def _exact_transform(grism_wcs):
    """Build the exact "detector" -> "grism_detector" (x, y) transform for comparison."""
    transform = grism_wcs.get_transform("detector", "grism_detector")
    n_outputs = len(transform.outputs)
    return transform | Mapping((0, 1), n_inputs=n_outputs)


def test_build_trace_lut_rejects_small_grid(grism_wcs):
    with pytest.raises(ValueError, match="n_grid"):
        build_trace_lut(grism_wcs, _ORDER, _WMIN, _WMAX, _NAXIS, n_grid=1)


def test_trace_lut_matches_exact_transform(grism_wcs):
    """Interpolated trace positions should closely match the exact transform off-grid."""
    trace_lut = build_trace_lut(grism_wcs, _ORDER, _WMIN, _WMAX, _NAXIS, n_grid=25)
    assert isinstance(trace_lut, TraceLUT)

    # Sample random points strictly inside the grid bounds, away from grid nodes.
    rng = np.random.default_rng(42)
    n_pts = 200
    x0 = rng.uniform(10, _NAXIS[0] - 10, n_pts)
    y0 = rng.uniform(10, _NAXIS[1] - 10, n_pts)
    wavelength = rng.uniform(_WMIN, _WMAX, n_pts)

    exact_transform = _exact_transform(grism_wcs)
    x_exact, y_exact = exact_transform(x0, y0, wavelength, _ORDER)
    x_interp, y_interp = trace_lut(x0, y0, wavelength)

    # Linear interpolation on a modest grid should recover sub-pixel accuracy
    # since the trace shape varies smoothly across the detector.
    assert_allclose(x_interp, x_exact, atol=0.5)
    assert_allclose(y_interp, y_exact, atol=0.5)


def test_trace_lut_preserves_shape(grism_wcs):
    """The TraceLUT should return outputs with the same shape as the inputs."""
    trace_lut = build_trace_lut(grism_wcs, _ORDER, _WMIN, _WMAX, _NAXIS, n_grid=10)

    x0 = np.full((5, 7), 500.0)
    y0 = np.full((5, 7), 600.0)
    wavelength = np.full((5, 7), 2.0)
    x, y = trace_lut(x0, y0, wavelength)
    assert x.shape == (5, 7)
    assert y.shape == (5, 7)


def test_evaluate_grid_matches_call(grism_wcs):
    """evaluate_grid should agree with the generic __call__ on the same query points."""
    trace_lut = build_trace_lut(grism_wcs, _ORDER, _WMIN, _WMAX, _NAXIS, n_grid=20)

    rng = np.random.default_rng(0)
    n_pixels = 37
    n_wave = 13
    x0 = rng.uniform(10, _NAXIS[0] - 10, n_pixels)
    y0 = rng.uniform(10, _NAXIS[1] - 10, n_pixels)
    wavelength = rng.uniform(_WMIN, _WMAX, n_wave)

    x_grid, y_grid = trace_lut.evaluate_grid(x0, y0, wavelength)
    assert x_grid.shape == (n_wave, n_pixels)
    assert y_grid.shape == (n_wave, n_pixels)

    # Build the same (n_wave, n_pixels) outer product manually and compare to __call__.
    x0_rep = np.repeat(x0[np.newaxis, :], n_wave, axis=0)
    y0_rep = np.repeat(y0[np.newaxis, :], n_wave, axis=0)
    lam_rep = np.repeat(wavelength[:, np.newaxis], n_pixels, axis=1)
    x_call, y_call = trace_lut(x0_rep, y0_rep, lam_rep)

    assert_allclose(x_grid, x_call, rtol=1e-10)
    assert_allclose(y_grid, y_call, rtol=1e-10)


def test_wavelength_grid_independent_of_spatial_n_grid(grism_wcs):
    """The wavelength grid density should not depend on the spatial n_grid parameter."""
    imgxy_to_grismxy = _exact_transform(grism_wcs)
    lam_grid = _native_wavelength_grid(
        imgxy_to_grismxy, _ORDER, _WMIN, _WMAX, _NAXIS[0] / 2.0, _NAXIS[1] / 2.0
    )
    # native spacing over this wavelength range should give many more points
    # than a typical small spatial grid, and should not equal it by coincidence
    for n_grid in (3, 10, 50):
        trace_lut = build_trace_lut(grism_wcs, _ORDER, _WMIN, _WMAX, _NAXIS, n_grid=n_grid)
        assert trace_lut._lam_grid.size == lam_grid.size
        assert trace_lut._x0_grid.size == n_grid
