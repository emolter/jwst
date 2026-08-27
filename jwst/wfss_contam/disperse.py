import logging
import multiprocessing as mp
import warnings

import numba as nb
import numpy as np
from astropy.modeling.mappings import Mapping
from scipy.interpolate import interp1d

from jwst.lib.winclip import get_clipped_pixels
from jwst.wfss_contam.sens1d import create_1d_sens

log = logging.getLogger(__name__)


@nb.njit(cache=True)
def _gather_by_index(lambdas_flat, fluxes_flat, source_ids_flat, index, lam_out, flux_out, sid_out):
    """
    Gather ``lambdas``, ``fluxes``, and ``source_ids`` at ``index`` in a single fused pass.

    Equivalent to calling ``np.take`` three times, but avoids doing three separate
    full-array gather passes over the (potentially large) input arrays.

    Parameters
    ----------
    lambdas_flat, fluxes_flat, source_ids_flat : np.ndarray
        Flattened 1-D input arrays to gather from.
    index : np.ndarray
        Flat indices into the input arrays, as returned by ``get_clipped_pixels``.
    lam_out, flux_out, sid_out : np.ndarray
        Pre-allocated output arrays of the same shape as ``index``, filled in place.
    """
    n = index.shape[0]
    for i in range(n):
        idx = index[i]
        lam_out[i] = lambdas_flat[idx]
        flux_out[i] = fluxes_flat[idx]
        sid_out[i] = source_ids_flat[idx]


__all__ = ["disperse"]


def _determine_native_wl_spacing(
    x0_sky,
    y0_sky,
    sky_to_imgxy,
    imgxy_to_grismxy,
    order,
    wmin,
    wmax,
    oversample_factor=2,
):
    """
    Determine the wavelength spacing necessary to adequately sample the dispersed frame.

    Parameters
    ----------
    x0_sky : float or ndarray
        RA of the input pixel position in direct image and segmentation map
    y0_sky : float or ndarray
        Dec of the input pixel position in direct image and segmentation map
    sky_to_imgxy : astropy model
        Transform from sky to image coordinates
    imgxy_to_grismxy : astropy model
        Transform from image to grism coordinates
    order : int
        Spectral order number
    wmin : float
        Minimum wavelength for dispersed spectra
    wmax : float
        Maximum wavelength for dispersed spectra
    oversample_factor : int, optional
        Factor by which to oversample the wavelength grid

    Returns
    -------
    lambdas : ndarray
        Wavelengths at which to compute dispersed pixel values

    Notes
    -----
    It was found that the native wavelength spacing varies by a few percent or less
    across the detector for both NIRCam and NIRISS. This function has the capability to
    take in many x0, y0 at once and take the median to get the wavelengths,
    but typically it's okay to just put in any x0, y0 pair.
    """
    # Get x/y positions in the grism image corresponding to wmin and wmax:
    # Convert to x/y in the direct image frame
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, order)
    # then convert to x/y in the grism image frame.
    xwmin, ywmin = imgxy_to_grismxy(x0_xy, y0_xy, wmin, order)
    xwmax, ywmax = imgxy_to_grismxy(x0_xy, y0_xy, wmax, order)
    dxw = xwmax - xwmin
    dyw = ywmax - ywmin

    # Create list of wavelengths on which to compute dispersed pixels
    dw = np.abs((wmax - wmin) / (dyw - dxw))
    dlam = np.median(dw / oversample_factor)
    # need at least three points because often the sensitivity curve
    # is not well-defined at the edges. This is typically hit only for Order 0,
    # since dlam can be large or poorly defined in that case.
    npts = max(int(np.ceil((wmax - wmin) / dlam)), 3)
    lambdas = np.linspace(wmin, wmax, npts)
    return lambdas


def _disperse_onto_grism(
    x0_sky, y0_sky, sky_to_imgxy, imgxy_to_grismxy, lambdas, order, trace_lut=None
):
    """
    Compute x/y positions in the grism image for the set of desired wavelengths.

    Parameters
    ----------
    x0_sky : ndarray
        RA of the input pixel position in direct image and segmentation map
    y0_sky : ndarray
        Dec of the input pixel position in direct image and segmentation map
    sky_to_imgxy : astropy model
        Transform from sky to image coordinates
    imgxy_to_grismxy : astropy model
        Transform from image to grism coordinates
    lambdas : ndarray
        Wavelengths at which to compute dispersed pixel values
    order : int
        Spectral order number
    trace_lut : `~jwst.wfss_contam.trace_lut.TraceLUT`, optional
        If provided, used in place of ``imgxy_to_grismxy`` to approximate the
        dispersed pixel positions via a cached, precomputed grid.

    Returns
    -------
    x0s : ndarray
        X coordinates of dispersed pixels in the grism image
    y0s : ndarray
        Y coordinates of dispersed pixels in the grism image
    lambdas : ndarray
        Wavelengths corresponding to each dispersed pixel
    """
    # sky_to_imgxy (the "world" to "detector" transform) is a purely geometric,
    # achromatic distortion: it does not depend on wavelength (as also assumed by
    # _determine_native_wl_spacing, which evaluates it with a placeholder wavelength
    # of 1). So evaluate it once on the unique per-pixel positions instead of on the
    # full (n_lam, n_pixels) outer product used further down, avoiding redundant work.
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, order)
    n_pixels = len(x0_xy)

    if trace_lut is not None:
        x0s, y0s = trace_lut.evaluate_grid(x0_xy, y0_xy, lambdas)
        lambdas = np.repeat(lambdas[:, np.newaxis], n_pixels, axis=1)
    else:
        n_lam = len(lambdas)
        x0_xy = np.repeat(x0_xy[np.newaxis, :], n_lam, axis=0)
        y0_xy = np.repeat(y0_xy[np.newaxis, :], n_lam, axis=0)
        lambdas = np.repeat(lambdas[:, np.newaxis], n_pixels, axis=1)
        x0s, y0s = imgxy_to_grismxy(x0_xy, y0_xy, lambdas, order)
    # x0s, y0s now have shape (n_lam, n_pixels)
    return x0s, y0s, lambdas


@nb.njit(cache=True)
def _group_bounds(xs, ys, group_idx, n_groups):
    """
    Compute per-group (per-source) min/max x, y bounds in a single pass over all pixels.

    Replaces ``np.minimum.reduceat``/``np.maximum.reduceat``, which require the input
    to already be sorted and grouped; this works on pixels in their original order.

    Parameters
    ----------
    xs, ys : np.ndarray
        Detector x, y position of each pixel.
    group_idx : np.ndarray
        Group (source) index of each pixel, shape ``(n_pixels,)``.
    n_groups : int
        Total number of groups (sources).

    Returns
    -------
    minxs, maxxs, minys, maxys : np.ndarray
        Per-group bounds, shape ``(n_groups,)``.
    """
    minxs = np.full(n_groups, xs[0])
    maxxs = np.full(n_groups, xs[0])
    minys = np.full(n_groups, ys[0])
    maxys = np.full(n_groups, ys[0])
    seen = np.zeros(n_groups, dtype=np.bool_)
    for i in range(xs.shape[0]):
        g = group_idx[i]
        x = xs[i]
        y = ys[i]
        if not seen[g]:
            minxs[g] = x
            maxxs[g] = x
            minys[g] = y
            maxys[g] = y
            seen[g] = True
        else:
            if x < minxs[g]:
                minxs[g] = x
            if x > maxxs[g]:
                maxxs[g] = x
            if y < minys[g]:
                minys[g] = y
            if y > maxys[g]:
                maxys[g] = y
    return minxs, maxxs, minys, maxys


@nb.njit(cache=True)
def _accumulate_by_group(xs, ys, values, group_idx, minxs, minys, widths, offsets, out):
    """
    Scatter-accumulate pixel values into per-group (per-source) output buffers.

    All groups' output images are packed into a single flat ``out`` buffer (one row
    per channel), at the offsets precomputed for each group. This does the work of
    calling ``np.bincount`` once per source in a single fused pass over every pixel,
    for every channel (flux plus any basis-model channels) at once.

    Parameters
    ----------
    xs, ys : np.ndarray
        Detector x, y position of each pixel, in any order.
    values : np.ndarray
        2-D array of shape ``(n_channels, n_pixels)`` with the per-pixel values to
        accumulate for each channel (e.g. flux, then one row per basis model).
    group_idx : np.ndarray
        Group (source) index of each pixel, shape ``(n_pixels,)``.
    minxs, minys : np.ndarray
        Per-group minimum x, y position, shape ``(n_groups,)``.
    widths : np.ndarray
        Per-group image width (in pixels), shape ``(n_groups,)``.
    offsets : np.ndarray
        Per-group starting offset into the flat ``out`` buffer, shape ``(n_groups,)``.
    out : np.ndarray
        2-D array of shape ``(n_channels, total_size)``, zeroed and filled in place.
    """
    n_pixels = xs.shape[0]
    n_channels = values.shape[0]
    for i in range(n_pixels):
        g = group_idx[i]
        local_idx = (ys[i] - minys[g]) * widths[g] + (xs[i] - minxs[g])
        flat_idx = offsets[g] + local_idx
        for c in range(n_channels):
            out[c, flat_idx] += values[c, i]


def _collect_outputs_by_source(xs, ys, counts, source_ids_per_pixel, model_counts=None):
    """
    Collect the dispersed pixel values into separate images for each source.

    Parameters
    ----------
    xs : ndarray
        X coordinates of dispersed pixels
    ys : ndarray
        Y coordinates of dispersed pixels
    counts : ndarray
        Count rates of dispersed pixels
    source_ids_per_pixel : int array
        Source IDs of the dispersed pixels. Must be non-negative (background/0 pixels
        are expected to already be filtered out upstream).
    model_counts : list of ndarray, optional
        List of count rate arrays corresponding to input ``basis_models``

    Returns
    -------
    outputs_by_source : dict
        Dictionary containing dispersed images and bounds for each source ID
    """
    if source_ids_per_pixel.size == 0:
        return {}
    has_models = model_counts is not None and len(model_counts) > 0

    # Map each pixel directly to a dense group (source) index without sorting: since
    # source IDs are small non-negative integers, np.bincount + cumsum over the (much
    # smaller) range of possible IDs is O(n_ids + n_pixels), avoiding the O(n_pixels *
    # log(n_pixels)) cost of np.argsort, and avoiding reordering xs/ys/counts at all.
    id_present = np.bincount(source_ids_per_pixel) > 0
    unique_ids = np.flatnonzero(id_present)
    n_groups = len(unique_ids)
    id_to_group = np.cumsum(id_present) - 1
    group_idx = id_to_group[source_ids_per_pixel]

    minxs, maxxs, minys, maxys = _group_bounds(xs, ys, group_idx, n_groups)
    widths = maxxs - minxs + 1
    heights = maxys - minys + 1
    sizes = widths * heights
    offsets = np.concatenate(([0], np.cumsum(sizes)))[:-1]
    total_size = int(sizes.sum())

    # Stack flux plus all basis-model channels so every source's image (and every
    # model_counts image) is accumulated in a single fused pass over all pixels.
    n_channels = 1 + (len(model_counts) if has_models else 0)
    values = np.empty((n_channels, len(xs)), dtype=counts.dtype)
    values[0] = counts
    if has_models:
        for k, mc in enumerate(model_counts):
            values[k + 1] = mc

    combined = np.zeros((n_channels, total_size), dtype=counts.dtype)
    _accumulate_by_group(xs, ys, values, group_idx, minxs, minys, widths, offsets, combined)

    outputs_by_source = {}
    for i, this_sid in enumerate(unique_ids):
        start = offsets[i]
        end = start + sizes[i]
        bounds = [int(minxs[i]), int(maxxs[i]), int(minys[i]), int(maxys[i])]
        outputs_by_source[this_sid] = {
            "bounds": bounds,
            "image": combined[0, start:end].reshape(heights[i], widths[i]),
        }
        if has_models:
            outputs_by_source[this_sid]["model_counts"] = [
                combined[k + 1, start:end].reshape(heights[i], widths[i])
                for k in range(len(model_counts))
            ]
    return outputs_by_source


def _replace_nans(fluxes):
    """
    Replace NaNs in multi-band fluxes along the wavelength axis (axis 0).

    Interior NaNs are filled by linear interpolation between the nearest valid
    bands on each side.  Edge NaNs (no valid band on one side) are filled by
    flat extrapolation from the nearest valid band.

    Parameters
    ----------
    fluxes : ndarray
        Array of shape (N, n_pixels) containing fluxes for N photometric bands.

    Returns
    -------
    filled_fluxes : ndarray
        Input array ``fluxes`` but with NaNs replaced, updated in place.
    """
    valid_mask = np.isfinite(fluxes)
    if not (~valid_mask).any():
        return fluxes

    n, _npix = fluxes.shape
    band_idx = np.arange(n)

    # For each position, find the index of the nearest valid band to the left
    # (or -1 if none) and to the right (or N if none) along wavelength axis (0).
    left_indices = np.where(valid_mask, band_idx[:, None], -1)
    np.maximum.accumulate(left_indices, axis=0, out=left_indices)

    right_indices = np.where(valid_mask, band_idx[:, None], n)
    np.minimum.accumulate(right_indices[::-1], axis=0, out=right_indices[::-1])

    # make bool arrays for whether there is a non-NaN band to the left or right of each NaN
    # rows is wavelength axis, cols is pixel axis
    nan_rows, nan_cols = np.where(~valid_mask)
    left_i = left_indices[nan_rows, nan_cols]
    right_i = right_indices[nan_rows, nan_cols]
    has_left = left_i >= 0
    has_right = right_i < n
    interior = has_left & has_right
    only_right = ~has_left & has_right
    only_left = has_left & ~has_right

    # interior NaNs: linearly interpolate
    if interior.any():
        r, c = nan_rows[interior], nan_cols[interior]
        # find flux at nearest non-nan to both left and right, then use those to find the slope
        li, ri = left_i[interior], right_i[interior]
        slope = (r - li) / (ri - li)
        fluxes[r, c] = fluxes[li, c] + slope * (fluxes[ri, c] - fluxes[li, c])

    # leading NaNs: flat fill from the right
    if only_right.any():
        r, c = nan_rows[only_right], nan_cols[only_right]
        # replace flux with that at nearest non-nan to the right
        fluxes[r, c] = fluxes[right_i[only_right], c]

    # trailing NaNs: flat fill from the left
    if only_left.any():
        r, c = nan_rows[only_left], nan_cols[only_left]
        # replace flux with that at nearest non-nan to the left
        fluxes[r, c] = fluxes[left_i[only_left], c]

    return fluxes


def disperse(
    xs,
    ys,
    fluxes,
    band_wavelengths,
    source_ids_per_pixel,
    order,
    wmin,
    wmax,
    sens_waves,
    sens_resp,
    direct_image_wcs,
    grism_wcs,
    naxis,
    oversample_factor=2,
    basis_models=None,
    trace_lut=None,
):
    """
    Compute the dispersed image pixel values from the direct image.

    Parameters
    ----------
    xs : ndarray
        Flat array of X coordinates of pixels in the direct image
    ys : ndarray
        Flat array of Y coordinates of pixels in the direct image
    fluxes : ndarray of shape (N, n_pixels)
        Fluxes of the pixels in the direct image corresponding to xs, ys,
        in units of MJy/sr.  N is the number of photometric bands; use N=1
        for a flat (wavelength-independent) SED. Note in that case the array must still be 2-D.
    band_wavelengths : ndarray
        Central wavelengths (in microns) of each photometric band in
        ``fluxes`` (shape (N,)).  Fluxes are linearly interpolated onto the internal
        wavelength grid. Fluxes are held constant (flat extrapolation)
        outside the covered wavelength range. For a flat SED this can be any length-1 array,
        as it is not used with N=1.
    source_ids_per_pixel : int array
        Source IDs of the input pixels in the segmentation map
    order : int
        Spectral order number
    wmin : float
        Minimum wavelength for dispersed spectra
    wmax : float
        Maximum wavelength for dispersed spectra
    sens_waves : float array
        Wavelength array from photom reference file. Expected unit is micron.
    sens_resp : float array
        Response (flux calibration) array from photom reference file.
        Expected units are (micron) * (MJy / sr) / (ADU/s).
    direct_image_wcs : WCS object
        WCS object for the direct image and segmentation map
    grism_wcs : WCS object
        WCS object for the grism image
    naxis : tuple
        Dimensions of the grism image (naxis[0], naxis[1])
    oversample_factor : int, optional
        Factor by which to oversample the wavelength grid
    basis_models : list[Callable], optional
        Flux distributions to evaluate at each wavelength. Typically these will be single
        polynomial orders, e.g. [lambda x: x, lambda x: x^2], ...] the coefficients of which
        are linearly fit later.
    trace_lut : `~jwst.wfss_contam.trace_lut.TraceLUT`, optional
        If provided, used in place of the exact "detector" to "grism_detector" transform
        to approximate the dispersed pixel positions via a cached, precomputed grid.
        This substantially speeds up dispersion at the cost of a small amount of accuracy.
        If None (the default), the exact transform is evaluated for every pixel.

    Returns
    -------
    outputs_by_source : dict
        Dictionary containing dispersed images and bounds for each source ID
        in the specified spectral order.
    """
    n_input_sources = np.unique(source_ids_per_pixel).size
    log.debug(
        f"{mp.current_process()} dispersing {n_input_sources} "
        f"sources in order {order} with total number of pixels: {len(xs)}"
    )
    width = 1.0
    height = 1.0
    x0 = xs + 0.5 * width
    y0 = ys + 0.5 * height
    del xs, ys

    # Set up the transforms we need from the input WCS objects
    sky_to_imgxy = grism_wcs.get_transform("world", "detector")
    imgxy_to_grismxy = grism_wcs.get_transform("detector", "grism_detector")

    # We only need the x,y outputs of imgxy_to_grismxy
    # Making the number of outputs dynamic handles legacy WCS objects that did not pass
    # the x0, y0, and order through the transform unmodified like the current version does.
    n_outputs = len(imgxy_to_grismxy.outputs)
    imgxy_to_grismxy = imgxy_to_grismxy | Mapping((0, 1), n_inputs=n_outputs)

    # Find RA/Dec of the input pixel position in direct image
    x0_sky, y0_sky = direct_image_wcs(x0, y0, with_bounding_box=False)
    del x0, y0

    # native spacing does not change much over the detector, so just put in one x0, y0
    lambdas = _determine_native_wl_spacing(
        x0_sky[0],
        y0_sky[0],
        sky_to_imgxy,
        imgxy_to_grismxy,
        order,
        wmin,
        wmax,
        oversample_factor=oversample_factor,
    )
    dlam = lambdas[1] - lambdas[0]
    nlam = len(lambdas)

    # Interpolate the input fluxes onto the wavelength grid of the dispersed image
    if len(band_wavelengths) >= 2:
        # interp1d does not handle NaNs, so replace with interplation that assumes
        # flat spectrum at the edges and linear interpolation in the interior,
        # which is what the behavior would be if we were to call interp1d separately
        # on each pixel's spectrum after removing NaNs.
        fluxes = _replace_nans(fluxes)
        interp_fn = interp1d(
            band_wavelengths,
            fluxes,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=(fluxes[0], fluxes[-1]),  # flat extrapolation
        )
        fluxes = interp_fn(lambdas)  # (nlam, n_pixels)
    else:
        # constant flux across all wavelengths
        fluxes = np.repeat(fluxes[0][np.newaxis, :], nlam, axis=0)
    source_ids_per_pixel = np.repeat(source_ids_per_pixel[np.newaxis, :], nlam, axis=0)

    x0s, y0s, lambdas = _disperse_onto_grism(
        x0_sky,
        y0_sky,
        sky_to_imgxy,
        imgxy_to_grismxy,
        lambdas,
        order,
        trace_lut=trace_lut,
    )
    del x0_sky, y0_sky

    # If none of the dispersed pixel indexes are within the image frame,
    # return a null result without wasting time doing other computations
    if x0s.min() >= naxis[0] or x0s.max() < 0 or y0s.min() >= naxis[1] or y0s.max() < 0:
        return

    # Discretize x and y coordinates to integer pixel values, keeping track of the fractional area
    # that each pixel contributes to the final grism image.
    # The resulting x, y coordinate pairs are non-unique: there are multiple wavelengths
    # that contribute to each pixel.
    padding = 1
    xs, ys, areas, index = get_clipped_pixels(x0s, y0s, padding, naxis[0], naxis[1], width, height)
    del x0s, y0s

    # Gather lambdas, fluxes, and source_ids_per_pixel at `index` in a single fused
    # pass instead of three separate np.take calls.
    lambdas_flat = lambdas.ravel()
    fluxes_flat = fluxes.ravel()
    source_ids_flat = source_ids_per_pixel.ravel()
    n_index = index.shape[0]
    lambdas = np.empty(n_index, dtype=lambdas_flat.dtype)
    fluxes = np.empty(n_index, dtype=fluxes_flat.dtype)
    source_ids_per_pixel = np.empty(n_index, dtype=source_ids_flat.dtype)
    _gather_by_index(
        lambdas_flat, fluxes_flat, source_ids_flat, index, lambdas, fluxes, source_ids_per_pixel
    )

    # Evaluate basis models on the 1-D lambda array.
    # even after gathering this is element-wise so this is still full resolution
    model_f = []
    if basis_models is not None:
        for flam in basis_models:
            model_f.append(flam(lambdas))

    # compute 1D sensitivity array corresponding to list of wavelengths
    sens, no_cal = create_1d_sens(lambdas, sens_waves, sens_resp)

    # Compute countrates for dispersed pixels.
    # The input direct image data is already photometrically calibrated,
    # so we need to basically apply a reverse flux calibration here.
    # Divide out the response values to convert from Mjy/sr to DN/s.
    # Note that the photom reference files are constructed with per-wavelength units,
    # so oversampling is accounted for by the spacing of dlam.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero|invalid value"
        )
        counts = fluxes * areas * dlam / sens
    counts[no_cal] = 0.0  # set to zero where no flux cal info available

    # Also convert basis models to counts.
    model_counts = []
    for f in model_f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="divide by zero|invalid value"
            )
            model_counts_i = fluxes * f * areas * dlam / sens
        model_counts_i[no_cal] = 0.0
        model_counts.append(model_counts_i)
    del fluxes, areas, sens, dlam, no_cal, lambdas, index

    outputs_by_source = _collect_outputs_by_source(
        xs, ys, counts, source_ids_per_pixel, model_counts
    )
    del xs, ys, counts, source_ids_per_pixel
    n_out = len(outputs_by_source)
    log.debug(
        f"{mp.current_process()} finished order {order} with {n_out} "
        f"sources that overlap with the output frame "
        f"(out of {n_input_sources} input sources)"
    )
    return outputs_by_source
