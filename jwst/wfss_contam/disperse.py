import logging
import multiprocessing as mp
from contextlib import nullcontext

import numpy as np
from astropy.modeling.mappings import Mapping
from scipy.interpolate import interp1d

from jwst.lib.winclip import get_clipped_pixels
from jwst.wfss_contam.pdt_fileio import open_pdt_part, write_pdt_to_files
from jwst.wfss_contam.sens1d import create_1d_sens

log = logging.getLogger(__name__)


__all__ = ["contam_model", "disperse_and_model", "disperse"]


def _disperse_onto_grism(
    x0_sky, y0_sky, sky_to_imgxy, imgxy_to_grismxy, lambdas, order, transform_interp=None
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
    transform_interp : \
        `~jwst.wfss_contam.transform_interp.DispersionTransformInterpolator`, optional
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
    # Evaluate sky-to-direct transform once on the unique per-pixel positions
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, order)
    n_pixels = len(x0_xy)

    if transform_interp is not None:
        x0s, y0s = transform_interp.evaluate_grid(x0_xy, y0_xy, lambdas)
        lambdas = np.repeat(lambdas[:, np.newaxis], n_pixels, axis=1)
    else:
        n_lam = len(lambdas)
        x0_xy = np.repeat(x0_xy[np.newaxis, :], n_lam, axis=0)
        y0_xy = np.repeat(y0_xy[np.newaxis, :], n_lam, axis=0)
        lambdas = np.repeat(lambdas[:, np.newaxis], n_pixels, axis=1)
        x0s, y0s = imgxy_to_grismxy(x0_xy, y0_xy, lambdas, order)
    # x0s, y0s now have shape (n_lam, n_pixels)
    return x0s, y0s, lambdas


def _collect_outputs_by_source(xs, ys, counts, source_ids, pixel_idx_ptr, model_counts=None):
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
    source_ids : ndarray
        Scalar source ID for a cached part, or source IDs per input pixel.
    pixel_idx_ptr : ndarray
        Contribution slice boundaries for the input pixels.
    model_counts : list of ndarray, optional
        List of count rate arrays corresponding to input ``basis_models``

    Returns
    -------
    outputs_by_source : dict
        Dictionary containing dispersed images and bounds for each source ID
    """
    model_counts = () if model_counts is None else model_counts
    if source_ids.ndim == 0:
        source_slices = [(source_ids.item(), slice(None))]
    else:
        if np.any(source_ids[1:] < source_ids[:-1]):
            contribution_ids = np.repeat(source_ids, np.diff(pixel_idx_ptr))
            sort_idx = np.argsort(contribution_ids, kind="stable")
            source_ids = contribution_ids[sort_idx]
            xs, ys, counts = xs[sort_idx], ys[sort_idx], counts[sort_idx]
            model_counts = [model[sort_idx] for model in model_counts]
            pixel_idx_ptr = np.arange(len(counts) + 1)

        starts = np.r_[0, np.flatnonzero(source_ids[1:] != source_ids[:-1]) + 1]
        boundaries = pixel_idx_ptr[np.r_[starts, len(source_ids)]]
        source_slices = zip(
            source_ids[starts],
            map(slice, boundaries[:-1], boundaries[1:]),
            strict=True,
        )

    outputs_by_source = {}
    for source_id, selection in source_slices:
        source_x, source_y = xs[selection], ys[selection]
        if not source_x.size:
            continue
        minx, maxx = int(source_x.min()), int(source_x.max())
        miny, maxy = int(source_y.min()), int(source_y.max())
        shape = (maxy - miny + 1, maxx - minx + 1)
        flat_idx = (source_y - miny) * shape[1] + (source_x - minx)
        result = {
            "bounds": [minx, maxx, miny, maxy],
            "image": _build_dispersed_image_of_source(flat_idx, counts[selection], shape),
        }
        if model_counts:
            result["model_counts"] = [
                _build_dispersed_image_of_source(flat_idx, model[selection], shape)
                for model in model_counts
            ]
        outputs_by_source[source_id] = result
    return outputs_by_source


def _build_dispersed_image_of_source(flat_idx, flux, shape):
    """
    Convert a flattened list of pixels to a 2-D grism image of that source.

    Parameters
    ----------
    flat_idx : ndarray
        Flattened pixel indices within the source cutout.
    flux : ndarray
        Fluxes of pixels in the grism image
    shape : tuple
        Cutout dimensions in row, column order.

    Returns
    -------
    ndarray
        2-D dispersed image of the source
    """
    counts = np.bincount(flat_idx, weights=flux, minlength=shape[0] * shape[1])
    return counts.reshape(shape).astype(flux.dtype, copy=False)


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
    source_ids_per_pixel,
    order,
    lambdas,
    direct_image_wcs,
    grism_wcs,
    naxis,
    transform_interp=None,
):
    """
    Compute the dispersed image pixel values from the direct image.

    Parameters
    ----------
    xs : ndarray
        Flat array of X coordinates of pixels in the direct image
    ys : ndarray
        Flat array of Y coordinates of pixels in the direct image
    source_ids_per_pixel : int array
        Source IDs of the input pixels in the segmentation map
    order : int
        Spectral order number
    lambdas : ndarray
        Wavelengths at which to compute dispersed pixel values.
    direct_image_wcs : WCS object
        WCS object for the direct image and segmentation map
    grism_wcs : WCS object
        WCS object for the grism image
    naxis : tuple
        Dimensions of the grism image (naxis[0], naxis[1])
    transform_interp : \
        `~jwst.wfss_contam.transform_interp.DispersionTransformInterpolator`, optional
        If provided, used in place of the exact "detector" to "grism_detector" transform
        to approximate the dispersed pixel positions via a cached, precomputed grid.
        This substantially speeds up dispersion at the cost of a small amount of accuracy.
        If None (the default), the exact transform is evaluated for every pixel.

    Returns
    -------
    dict
        Dictionary containing a slitlessutils-like pixel dispersion table.
    """
    input_xy = np.column_stack((xs, ys))
    source_ids_per_pixel = np.asarray(source_ids_per_pixel)
    n_input_sources = np.unique(source_ids_per_pixel).size
    log.debug(
        f"{mp.current_process()} dispersing {n_input_sources} "
        f"sources in order {order} with total number of pixels: {len(xs)}"
    )
    if source_ids_per_pixel.size == 0:
        return {
            "input_xy": input_xy,
            "source_id": source_ids_per_pixel,
            "pixel_idx_ptr": np.zeros(1, dtype=np.intp),
            "grism_xy": np.empty((0, 2), dtype=int),
            "wavelength_index": np.empty(0, dtype=int),
            "fractional_area": np.empty(0, dtype=float),
        }

    if np.max(source_ids_per_pixel) < np.iinfo(np.uint16).max and np.min(source_ids_per_pixel) >= 0:
        # Conversion to 16-bit makes the array faster to sort later
        source_ids_per_pixel = source_ids_per_pixel.astype(np.uint16, copy=False)
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

    nlam = len(lambdas)
    # Keep 1-d versions of lambdas, source_ids_per_pixel around so the gather below
    # can index based on these smaller arrays instead of the (nlam, n_pixels) broadcasts.
    n_pixels = len(source_ids_per_pixel)
    source_ids_1d = source_ids_per_pixel

    x0s, y0s, lambdas = _disperse_onto_grism(
        x0_sky,
        y0_sky,
        sky_to_imgxy,
        imgxy_to_grismxy,
        lambdas,
        order,
        transform_interp=transform_interp,
    )
    del x0_sky, y0_sky

    # If none of the dispersed pixel indexes are within the image frame,
    # return a null result without wasting time doing other computations
    possible = (x0s > -0.5) & (x0s < naxis[0] + 0.5) & (y0s > -0.5) & (y0s < naxis[1] + 0.5)
    if not possible.any():
        return {
            "input_xy": input_xy,
            "source_id": source_ids_1d,
            "pixel_idx_ptr": np.zeros(n_pixels + 1, dtype=np.intp),
            "grism_xy": np.empty((0, 2), dtype=int),
            "wavelength_index": np.empty(0, dtype=int),
            "fractional_area": np.empty(0, dtype=float),
        }

    # Discretize x and y coordinates to integer pixel values, keeping track of the fractional area
    # that each pixel contributes to the final grism image.
    # The resulting x, y coordinate pairs are non-unique: there are multiple wavelengths
    # that contribute to each pixel.
    padding = 1
    xs, ys, areas, index = get_clipped_pixels(
        np.ascontiguousarray(x0s.T),
        np.ascontiguousarray(y0s.T),
        padding,
        naxis[0],
        naxis[1],
        width,
        height,
    )
    del x0s, y0s

    pixel_idx, lam_idx = np.divmod(index, nlam)
    pixel_idx_ptr = np.empty(n_pixels + 1, dtype=np.intp)
    pixel_idx_ptr[0] = 0
    np.cumsum(np.bincount(pixel_idx, minlength=n_pixels), out=pixel_idx_ptr[1:])

    # this is basically a slitlessutils-style Pixel Dispersion Table (PDT)
    # pixel_idx_ptr is an array of length n_pixels + 1 that points to the start of each pixel's
    # contributions in the flattened arrays grism_xy, wavelength_index, and fractional_area.
    geometry = {
        "input_xy": input_xy,
        "source_id": source_ids_1d,
        "pixel_idx_ptr": pixel_idx_ptr,
        "grism_xy": np.column_stack((xs, ys)),
        "wavelength_index": lam_idx,
        "fractional_area": areas,
    }
    return geometry


def contam_model(
    pdt,
    fluxes,
    band_wavelengths,
    sens_waves,
    sens_resp,
    basis_models=None,
):
    """
    Compute the dispersed image pixel values based on the direct image fluxes.

    Parameters
    ----------
    pdt : dict
        Pre-computed table storing mapping between direct image pixels and
        corresponding dispersed position, wavelength, and fractional area.
    fluxes : ndarray
        Direct image fluxes in MJy/sr, with shape (N, n_pixels), aligned
        with the input pixels in the PDT. Use N=1 for a flat SED.
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
    wavelength_index = np.asarray(pdt["wavelength_index"])
    if wavelength_index.size == 0:
        return {}

    lambdas = np.asarray(pdt["wavelength"], dtype=float)
    nlam = lambdas.size
    dlam = (lambdas[-1] - lambdas[0]) / (nlam - 1)
    # Interpolate the input fluxes onto the wavelength grid of the dispersed image
    if len(band_wavelengths) >= 2:
        # interp1d does not handle NaNs, so replace with interplation that assumes
        # flat spectrum at the edges and linear interpolation in the interior,
        # which is what the behavior would be if we were to call interp1d separately
        # on each pixel's spectrum after removing NaNs.
        fluxes = _replace_nans(fluxes.copy())
        interp_fn = interp1d(
            band_wavelengths,
            fluxes,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=(fluxes[0], fluxes[-1]),  # flat extrapolation
        )
        fluxes = interp_fn(lambdas)  # (nlam, n_pixels)
    # compute 1D sensitivity array corresponding to list of wavelengths
    sens, no_cal = create_1d_sens(lambdas, sens_waves, sens_resp)

    pixel_idx_ptr = np.asarray(pdt["pixel_idx_ptr"], dtype=np.intp)
    contributions_per_pixel = np.diff(pixel_idx_ptr)
    input_pixel_index = np.repeat(np.arange(len(pdt["input_xy"])), contributions_per_pixel)
    if len(band_wavelengths) >= 2:
        contribution_flux = fluxes[wavelength_index, input_pixel_index]
    else:
        contribution_flux = fluxes[0, input_pixel_index]
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
    source_ids = np.asarray(pdt["source_id"])
    return _collect_outputs_by_source(xs, ys, counts, source_ids, pixel_idx_ptr, model_counts)


def disperse_and_model(
    xs,
    ys,
    fluxes,
    band_wavelengths,
    source_ids_per_pixel,
    order,
    lambdas,
    sens_waves,
    sens_resp,
    direct_image_wcs,
    grism_wcs,
    naxis,
    basis_models=None,
    transform_interp=None,
    pdt_directory=None,
    source_parts=None,
    cached_part=None,
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
    lambdas : ndarray
        Wavelengths at which to compute dispersed pixel values.
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
    basis_models : list[Callable], optional
        Flux distributions to evaluate at each wavelength. Typically these will be single
        polynomial orders, e.g. [lambda x: x, lambda x: x^2], ...] the coefficients of which
        are linearly fit later.
    transform_interp : \
        `~jwst.wfss_contam.transform_interp.DispersionTransformInterpolator`, optional
        If provided, used in place of the exact "detector" to "grism_detector" transform
        to approximate the dispersed pixel positions via a cached, precomputed grid.
        This substantially speeds up dispersion at the cost of a small amount of accuracy.
        If None (the default), the exact transform is evaluated for every pixel.
    pdt_directory : str or Path, optional
        Directory where the PDT files will be written, or None to skip writing.
    source_parts : dict, optional
        Source IDs mapped to part numbers when writing to disk.
    cached_part : tuple, optional
        Source ID and path of a saved part. When provided, fluxes must be aligned
        with its stored input pixels and no WCS transforms are evaluated.

    Returns
    -------
    outputs_by_source : dict
        Dictionary containing dispersed images and bounds for each source ID
        in the specified spectral order.
        When saving, returned together with the written part metadata as a tuple.
    """
    if cached_part is None:
        geometry = disperse(
            xs,
            ys,
            source_ids_per_pixel,
            order,
            lambdas,
            direct_image_wcs,
            grism_wcs,
            naxis,
            transform_interp=transform_interp,
        )
        geometry["wavelength"] = lambdas
        geometry["order"] = order
        if pdt_directory is not None:
            written_parts = write_pdt_to_files(geometry, pdt_directory, source_parts)
        geometry_context = nullcontext(geometry)
    else:
        source_id, path = cached_part
        geometry_context = open_pdt_part(path, order, source_id)

    with geometry_context as geometry:
        outputs_by_source = contam_model(
            dict(geometry, wavelength=lambdas),
            fluxes,
            band_wavelengths,
            sens_waves,
            sens_resp,
            basis_models=basis_models,
        )
    if pdt_directory is not None and cached_part is None:
        return outputs_by_source, written_parts
    return outputs_by_source
