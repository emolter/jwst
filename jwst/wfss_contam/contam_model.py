import logging

import numpy as np
from scipy.interpolate import interp1d

from jwst.wfss_contam.sens1d import create_1d_sens

log = logging.getLogger(__name__)


__all__ = ["contam_model"]


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
        Source IDs of the dispersed pixels
    model_counts : list of ndarray, optional
        List of count rate arrays corresponding to input ``basis_models``

    Returns
    -------
    outputs_by_source : dict
        Dictionary containing dispersed images and bounds for each source ID
    """
    # Chunks are pre-sorted by source ID before dispersing (see Observation.chunk_sources).
    # Many observations have sources large enough that some chunks contain only pixels from
    # a single source. The collect operation can be simplified a lot in that case.
    first_sid = source_ids_per_pixel[0]
    if (source_ids_per_pixel == first_sid).all():
        bounds = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
        img = _build_dispersed_image_of_source(xs, ys, counts, bounds)
        outputs_by_source = {first_sid: {"bounds": bounds, "image": img}}
        if model_counts is not None and len(model_counts) > 0:
            outputs_by_source[first_sid]["model_counts"] = [
                _build_dispersed_image_of_source(xs, ys, mc, bounds) for mc in model_counts
            ]
        return outputs_by_source

    # First sort by source ID. xs, ys input here cannot be assumed sorted after get_clipped_pixels
    sort_idx = np.argsort(source_ids_per_pixel, kind="stable")
    sorted_ids = source_ids_per_pixel[sort_idx]
    sorted_xs = xs[sort_idx]
    sorted_ys = ys[sort_idx]
    sorted_counts = counts[sort_idx]
    if model_counts is not None and len(model_counts) > 0:
        sorted_model_counts = [mc[sort_idx] for mc in model_counts]

    # Compute per-source bounds in a vectorized way
    # The following five lines used to be just
    # `unique_ids, split_points = np.unique(sorted_ids, return_index=True)`
    # but that re-sorts the array. This is more verbose but faster.
    is_boundary = np.empty(len(sorted_ids), dtype=bool)
    is_boundary[0] = True
    np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=is_boundary[1:])
    split_points = np.flatnonzero(is_boundary)
    unique_ids = sorted_ids[split_points]

    minxs = np.minimum.reduceat(sorted_xs, split_points)
    maxxs = np.maximum.reduceat(sorted_xs, split_points)
    minys = np.minimum.reduceat(sorted_ys, split_points)
    maxys = np.maximum.reduceat(sorted_ys, split_points)

    # Now loop through sources, build the output images, and store bounds
    # to reconstruct the full dispersed image later
    outputs_by_source = {}
    for i, this_sid in enumerate(unique_ids):
        start = split_points[i]
        end = split_points[i + 1] if i + 1 < len(split_points) else len(sorted_xs)
        this_xs = sorted_xs[start:end]
        this_ys = sorted_ys[start:end]
        this_flxs = sorted_counts[start:end]

        bounds = [int(minxs[i]), int(maxxs[i]), int(minys[i]), int(maxys[i])]
        img = _build_dispersed_image_of_source(this_xs, this_ys, this_flxs, bounds)
        outputs_by_source[this_sid] = {
            "bounds": bounds,
            "image": img,
        }
        if model_counts is not None and len(model_counts) > 0:
            outputs_by_source[this_sid]["model_counts"] = [
                _build_dispersed_image_of_source(this_xs, this_ys, mc[start:end], bounds)
                for mc in sorted_model_counts
            ]
    return outputs_by_source


def _build_dispersed_image_of_source(x, y, flux, bounds):
    """
    Convert a flattened list of pixels to a 2-D grism image of that source.

    Parameters
    ----------
    x : ndarray
        X coordinates of pixels in the grism image
    y : ndarray
        Y coordinates of pixels in the grism image
    flux : ndarray
        Fluxes of pixels in the grism image
    bounds : list
        Pre-computed [minx, maxx, miny, maxy] bounds for the source.

    Returns
    -------
    a : ndarray
        2-D dispersed image of the source

    Notes
    -----
    This function used to read as
        img = np.zeros((maxy - miny + 1, maxx - minx + 1), dtype=flux.dtype)
        np.add.at(img, (y - miny, x - minx), flux)
        return img
    but bincount is much faster albeit less readable, since we need to cast to 1-d
    and then cast back to 2-d at the end.
    """
    minx, maxx, miny, maxy = bounds
    ncols = maxx - minx + 1
    nrows = maxy - miny + 1

    # Cast into 1-D so we can use bincount, which is faster than np.add.at
    flat_idx = (y - miny) * ncols + (x - minx)
    counts = np.bincount(flat_idx, weights=flux, minlength=nrows * ncols)
    return counts.reshape((nrows, ncols)).astype(flux.dtype, copy=False)


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


def contam_model(
    pdt,
    fluxes,
    band_wavelengths,
    sens_waves,
    sens_resp,
    basis_models=None,
):
    """
    Compute the dispersed image pixel values from the direct image.

    Parameters
    ----------
    pdt : dict
        Pre-computed table storing mapping between direct image pixels and
        corresponding dispersed position, wavelength, and fractional area.
    fluxes : ndarray
        Direct image fluxes in MJy/sr, with shape (ny, nx) for a flat SED
        or (N, ny, nx) for N photometric bands.
    band_wavelengths : ndarray
        Central wavelengths (in microns) of each photometric band in
        ``fluxes`` (shape (N,)).  Fluxes are linearly interpolated onto the internal
        wavelength grid. Fluxes are held constant (flat extrapolation)
        outside the covered wavelength range. For a flat SED this can be any length-1 array,
        as it is not used with N=1.
    sens_waves : float array
        Wavelength array from photom reference file. Expected unit is micron.
    sens_resp : float array
        Response (flux calibration) array from photom reference file.
        Expected units are (micron) * (MJy / sr) / (ADU/s).
    basis_models : list[Callable], optional
        Flux distributions to evaluate at each wavelength. Typically these will be single
        polynomial orders, e.g. [lambda x: x, lambda x: x^2], ...] the coefficients of which
        are linearly fit later.

    Returns
    -------
    outputs_by_source : dict
        Dictionary containing dispersed images and bounds for each source ID
        in the specified spectral order.
    """
    wavelength_index = np.asarray(pdt["wavelength_index"], dtype=np.intp)
    if wavelength_index.size == 0:
        return {}

    input_x, input_y = np.asarray(pdt["input_xy"], dtype=np.intp).T
    if fluxes.ndim == 2:
        fluxes = fluxes[np.newaxis, ...]
    fluxes = fluxes[:, input_y, input_x].copy()

    lambdas = np.asarray(pdt["wavelength"], dtype=float)
    nlam = lambdas.size
    dlam = (lambdas[-1] - lambdas[0]) / (nlam - 1)
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

    # compute 1D sensitivity array corresponding to list of wavelengths
    sens, no_cal = create_1d_sens(lambdas, sens_waves, sens_resp)

    contributions_per_pixel = np.diff(np.asarray(pdt["pixel_idx_ptr"], dtype=np.intp))
    input_pixel_index = np.repeat(np.arange(len(pdt["input_xy"])), contributions_per_pixel)
    contribution_flux = fluxes[wavelength_index, input_pixel_index]
    fractional_area = np.asarray(pdt["fractional_area"], dtype=float)

    # Compute countrates for dispersed pixels.
    # The input direct image data is already photometrically calibrated,
    # so we need to basically apply a reverse flux calibration here.
    # Divide out the response values to convert from Mjy/sr to DN/s.
    # Note that the photom reference files are constructed with per-wavelength units,
    # so oversampling is accounted for by the spacing of dlam.
    conversion = np.zeros_like(sens, dtype=float)
    np.divide(dlam, sens, out=conversion, where=~no_cal)
    counts = contribution_flux * fractional_area * conversion[wavelength_index]
    counts[no_cal[wavelength_index]] = 0.0

    # Also convert basis models to counts.
    model_counts = []
    if basis_models is not None:
        for basis_model in basis_models:
            basis_values = np.broadcast_to(np.asarray(basis_model(lambdas)), lambdas.shape)
            basis_counts = counts * basis_values[wavelength_index]
            basis_counts[no_cal[wavelength_index]] = 0.0
            model_counts.append(basis_counts)

    xs, ys = np.asarray(pdt["grism_xy"], dtype=np.intp).T
    source_ids_per_contribution = np.broadcast_to(int(pdt["source_id"]), wavelength_index.shape)
    return _collect_outputs_by_source(xs, ys, counts, source_ids_per_contribution, model_counts)
