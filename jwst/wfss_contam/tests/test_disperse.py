import numpy as np
from astropy.modeling.mappings import Mapping
from numpy.testing import assert_allclose

from jwst.wfss_contam.disperse import (
    _collect_outputs_by_source,
    _disperse_onto_grism,
    _replace_nans,
    disperse,
)
from jwst.wfss_contam.trace_lut import build_trace_lut

_SENS_WAVES = np.linspace(1.708, 2.28, 100)
_WMIN, _WMAX = _SENS_WAVES[0], _SENS_WAVES[-1]
_NAXIS = (300, 500)
_SOURCE_ID = 50


def test_sky_to_imgxy_independent_of_wavelength(grism_wcs, direct_image_with_gradient):
    """
    Guard the assumption that "world" -> "detector" does not depend on wavelength.

    _disperse_onto_grism relies on this to avoid evaluating that transform on the
    full (n_lam, n_pixels) outer product; if this assumption ever breaks for some
    reference file, this test should catch it.
    """
    direct_image_wcs = direct_image_with_gradient.meta.wcs
    sky_to_imgxy = grism_wcs.get_transform("world", "detector")
    imgxy_to_grismxy = grism_wcs.get_transform("detector", "grism_detector")
    n_outputs = len(imgxy_to_grismxy.outputs)
    imgxy_to_grismxy = imgxy_to_grismxy | Mapping((0, 1), n_inputs=n_outputs)

    x0 = np.array([100.5, 200.5, 300.5])
    y0 = np.array([100.5, 150.5, 250.5])
    x0_sky, y0_sky = direct_image_wcs(x0, y0, with_bounding_box=False)

    lambdas = np.linspace(_WMIN, _WMAX, 7)
    x0_ref, y0_ref, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, 1)

    for lam in lambdas:
        x0_lam, y0_lam, _, _ = sky_to_imgxy(x0_sky, y0_sky, lam, 1)
        assert_allclose(x0_lam, x0_ref)
        assert_allclose(y0_lam, y0_ref)

    # sanity check the function under test still returns the expected shape
    x0s, y0s, lambdas_out = _disperse_onto_grism(
        x0_sky, y0_sky, sky_to_imgxy, imgxy_to_grismxy, lambdas, 1
    )
    assert x0s.shape == (len(lambdas), len(x0))
    assert y0s.shape == (len(lambdas), len(x0))
    assert lambdas_out.shape == (len(lambdas), len(x0))


def test_disperse_with_trace_lut_matches_exact(grism_wcs, direct_image_with_gradient):
    """Dispersing with a trace_lut should closely match the exact per-pixel transform."""
    x0 = np.array([200.5])
    y0 = np.array([200.5])
    order = 1
    flxs = np.array([[1.0]])
    band_wave = np.array([2.0])
    source_id = np.array([50])
    direct_image_wcs = direct_image_with_gradient.meta.wcs

    trace_lut = build_trace_lut(grism_wcs, order, _WMIN, _WMAX, _NAXIS, n_grid=25)

    common_kwargs = {
        "fluxes": flxs,
        "band_wavelengths": band_wave,
        "source_ids_per_pixel": source_id,
        "order": order,
        "wmin": _WMIN,
        "wmax": _WMAX,
        "sens_waves": _SENS_WAVES,
        "sens_resp": np.ones(100),
        "direct_image_wcs": direct_image_wcs,
        "grism_wcs": grism_wcs,
        "naxis": _NAXIS,
    }
    exact = disperse(x0, y0, trace_lut=None, **common_kwargs)
    approx = disperse(x0, y0, trace_lut=trace_lut, **common_kwargs)

    img_exact = exact[_SOURCE_ID]["image"]
    img_approx = approx[_SOURCE_ID]["image"]
    assert img_exact.shape == img_approx.shape
    # the trace_lut trades a small amount of accuracy for speed, so allow some tolerance
    assert_allclose(img_approx.sum(), img_exact.sum(), rtol=0.05)


def test_disperse_oversample_same_result(grism_wcs, direct_image_with_gradient):
    """Coverage for bug where wavelength oversampling led to double-counted fluxes."""
    x0 = np.array([200.5])
    y0 = np.array([200.5])
    order = 1
    flxs = np.array([[1.0]])
    band_wave = np.array([2.0])
    source_id = np.array([50])
    naxis = (300, 500)
    sens_waves = np.linspace(1.708, 2.28, 100)
    wmin, wmax = np.min(sens_waves), np.max(sens_waves)
    sens_resp = np.ones(100)
    direct_image_wcs = direct_image_with_gradient.meta.wcs

    output_images = []
    for os in [2, 3]:
        src = disperse(
            x0,
            y0,
            flxs,
            band_wave,
            source_id,
            order,
            wmin,
            wmax,
            sens_waves,
            sens_resp,
            direct_image_wcs,
            grism_wcs,
            naxis,
            oversample_factor=os,
        )
        output_images.append(src[source_id[0]]["image"])

    # different oversampling gives different effects at the ends
    # unsure if this is a bug or not, but the middle should definitely be the same
    assert_allclose(output_images[0][2:-2, :], output_images[1][2:-2], rtol=1e-5)
    assert "model_counts" not in src[source_id[0]]


def _disperse_one_pixel(fluxes, band_wavelengths, grism_wcs, direct_image_wcs, sens_resp=None):
    """Run disperse() for a single pixel with unit sensitivity, returning the output image."""
    x0 = np.array([200.5])
    y0 = np.array([200.5])
    source_id = np.array([50])
    naxis = (300, 500)
    sens_waves = np.linspace(1.708, 2.28, 100)
    wmin, wmax = float(sens_waves[0]), float(sens_waves[-1])
    if sens_resp is None:
        sens_resp = np.ones(100)
    src = disperse(
        x0,
        y0,
        fluxes,
        band_wavelengths,
        source_id,
        order=1,
        wmin=wmin,
        wmax=wmax,
        sens_waves=sens_waves,
        sens_resp=sens_resp,
        direct_image_wcs=direct_image_wcs,
        grism_wcs=grism_wcs,
        naxis=naxis,
    )
    return src[source_id[0]]["image"]


def test_disperse_n1_matches_n2_constant(grism_wcs, direct_image_with_gradient):
    """Test that the single-band case and the interp1d path agree when the SED is flat."""
    direct_image_wcs = direct_image_with_gradient.meta.wcs
    img_n1 = _disperse_one_pixel(
        fluxes=np.array([[1.0]]),
        band_wavelengths=np.array([2.0]),
        grism_wcs=grism_wcs,
        direct_image_wcs=direct_image_wcs,
    )
    img_n2 = _disperse_one_pixel(
        fluxes=np.array([[1.0], [1.0]]),
        band_wavelengths=np.array([1.708, 2.28]),
        grism_wcs=grism_wcs,
        direct_image_wcs=direct_image_wcs,
    )
    assert_allclose(img_n1, img_n2, rtol=1e-10)


def test_disperse_flux_extrapolation(grism_wcs, direct_image_with_gradient):
    """Test that extrapolation is flat outside the band coverage."""
    direct_image_wcs = direct_image_with_gradient.meta.wcs
    sens_waves = np.linspace(1.708, 2.28, 100)
    wmin, wmax = float(sens_waves[0]), float(sens_waves[-1])
    wmid1 = wmin + 0.3 * (wmax - wmin)
    wmid2 = wmax - 0.3 * (wmax - wmin)

    img_extrap = _disperse_one_pixel(
        fluxes=np.array([[0.0], [2.0]]),
        band_wavelengths=np.array([wmid1, wmid2]),
        grism_wcs=grism_wcs,
        direct_image_wcs=direct_image_wcs,
    )
    # manual 4-plane SED that encodes the same flat extrapolation
    img_explicit = _disperse_one_pixel(
        fluxes=np.array([[0.0], [0.0], [2.0], [2.0]]),
        band_wavelengths=np.array([wmin, wmid1, wmid2, wmax]),
        grism_wcs=grism_wcs,
        direct_image_wcs=direct_image_wcs,
    )
    assert_allclose(img_extrap, img_explicit, rtol=1e-10)


def test_disperse_flux_distribution(grism_wcs, direct_image_with_gradient):
    """
    Test that multi-band input is being accounted for.

    Use a sensitivity curve that passes only the lower half of the wavelength range,
    so only the short wavelengths contribute to the output. Then test two SEDs
    with the same average flux but different wavelength dependence.
    """
    direct_image_wcs = direct_image_with_gradient.meta.wcs
    sens_waves = np.linspace(1.708, 2.28, 100)
    wmin, wmax = float(sens_waves[0]), float(sens_waves[-1])

    # Sensitivity passes only the lower half of the wavelength range
    # This allows us to test that the interpolation is actually doing something
    sens_resp = np.zeros(100)
    sens_resp[:50] = 1.0

    bw = np.array([wmin, wmax])
    sum_falling = _disperse_one_pixel(
        np.array([[2.0], [0.0]]),
        bw,
        grism_wcs=grism_wcs,
        direct_image_wcs=direct_image_wcs,
        sens_resp=sens_resp,
    ).sum()
    sum_rising = _disperse_one_pixel(
        np.array([[0.0], [2.0]]),
        bw,
        grism_wcs=grism_wcs,
        direct_image_wcs=direct_image_wcs,
        sens_resp=sens_resp,
    ).sum()

    assert sum_falling / sum_rising > 2.0


def _disperse_one_pixel_with_basis(grism_wcs, direct_image_wcs, basis_models=None):
    """Run disperse() for a single pixel with flat unit sensitivity."""
    xs = np.array([200])
    ys = np.array([200])
    fluxes = np.array([[1.0]])
    band_wavelengths = np.array([2.0])
    source_ids = np.array([_SOURCE_ID])
    sens_resp = np.ones_like(_SENS_WAVES)
    return disperse(
        xs,
        ys,
        fluxes,
        band_wavelengths,
        source_ids,
        1,
        _WMIN,
        _WMAX,
        _SENS_WAVES,
        sens_resp,
        direct_image_wcs,
        grism_wcs,
        _NAXIS,
        basis_models=basis_models,
    )


def test_flux_models_output_structure(grism_wcs, direct_image_with_gradient):
    """Test that source dict has a model_counts list with one entry per input model."""
    models = [lambda x: x, lambda x: x**2]
    src = _disperse_one_pixel_with_basis(
        grism_wcs, direct_image_with_gradient.meta.wcs, basis_models=models
    )
    assert "model_counts" in src[_SOURCE_ID]
    assert len(src[_SOURCE_ID]["model_counts"]) == 2
    for mc in src[_SOURCE_ID]["model_counts"]:
        assert mc.shape == src[_SOURCE_ID]["image"].shape


def test_flux_models_scalar_scaling(grism_wcs, direct_image_with_gradient):
    """Test that scaling a flux model scales the output model_counts image."""
    wcs = direct_image_with_gradient.meta.wcs
    src1 = _disperse_one_pixel_with_basis(grism_wcs, wcs, basis_models=[lambda x: x])
    src3 = _disperse_one_pixel_with_basis(grism_wcs, wcs, basis_models=[lambda x: 3.0 * x])
    assert_allclose(src3[_SOURCE_ID]["model_counts"][0], 3.0 * src1[_SOURCE_ID]["model_counts"][0])


def test_flux_models_superposition(grism_wcs, direct_image_with_gradient):
    """
    The sum of two separate model images should equal the image from their combined model.

    This somewhat explains why using basis functions this way is mathematically valid.
    If the basis models are being linearly combined, it doesn't matter if they are combined
    before or after discretizing the dispersed image onto a pixel grid.
    """
    wcs = direct_image_with_gradient.meta.wcs
    f1 = lambda x: x
    f2 = lambda x: x**2
    fsum = lambda x: f1(x) + f2(x)

    src_sep = _disperse_one_pixel_with_basis(grism_wcs, wcs, basis_models=[f1, f2])
    src_sum = _disperse_one_pixel_with_basis(grism_wcs, wcs, basis_models=[fsum])

    combined = src_sep[_SOURCE_ID]["model_counts"][0] + src_sep[_SOURCE_ID]["model_counts"][1]
    assert_allclose(combined, src_sum[_SOURCE_ID]["model_counts"][0], rtol=1e-10)


def test_collect_outputs_by_source_accumulates_duplicates():
    """Duplicate coordinates within a source should sum their flux contributions."""
    result = _collect_outputs_by_source(
        xs=np.array([0, 0, 1]),
        ys=np.array([0, 0, 1]),
        counts=np.array([1.0, 2.0, 3.0]),
        source_ids_per_pixel=np.array([7, 7, 7]),
    )
    assert_allclose(result[7]["image"], np.array([[3.0, 0.0], [0.0, 3.0]]))


def test_collect_outputs_by_source_empty():
    """No input pixels should return an empty dict rather than raise."""
    empty = np.array([], dtype=int)
    result = _collect_outputs_by_source(empty, empty, np.array([]), empty)
    assert result == {}


def test_collect_outputs_by_source_groups_correctly():
    """Unsorted, interleaved source IDs should be grouped into separate images/bounds."""
    xs = np.array([5, 0, 5, 1])
    ys = np.array([5, 0, 6, 1])
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    source_ids = np.array([2, 1, 2, 1])

    result = _collect_outputs_by_source(xs, ys, counts, source_ids)

    assert set(result.keys()) == {1, 2}
    assert result[1]["bounds"] == [0, 1, 0, 1]
    assert_allclose(result[1]["image"], np.array([[2.0, 0.0], [0.0, 4.0]]))
    assert result[2]["bounds"] == [5, 5, 5, 6]
    assert_allclose(result[2]["image"], np.array([[1.0], [3.0]]))


def test_replace_nans():
    """Test that _replace_nans() does what we expect."""
    arr = np.array([[np.nan, np.nan, 3.0, np.nan, 5.0, np.nan]]).T
    arr = _replace_nans(arr)
    assert_allclose(arr, np.array([[3.0, 3.0, 3.0, 4.0, 5.0, 5.0]]).T)


def test_replace_nans_no_nans():
    """Test that _replace_nans() does nothing if there are no NaNs."""
    arr = np.array([[1.0, 2.0, 3.0]]).T
    filled = _replace_nans(arr)
    assert filled is arr
