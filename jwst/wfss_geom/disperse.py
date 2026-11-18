import logging
import multiprocessing as mp

import numpy as np
from astropy.modeling.mappings import Mapping

from jwst.lib.winclip import get_clipped_pixels

log = logging.getLogger(__name__)


__all__ = ["disperse"]


def _disperse_onto_grism(
    x0_sky, y0_sky, sky_to_imgxy, imgxy_to_grismxy, lambdas, order, trace_pdt=None
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
    trace_pdt : `~jwst.wfss_contam.trace_pdt.TracePDT`, optional
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

    if trace_pdt is not None:
        x0s, y0s = trace_pdt.evaluate_grid(x0_xy, y0_xy, lambdas)
        lambdas = np.repeat(lambdas[:, np.newaxis], n_pixels, axis=1)
    else:
        n_lam = len(lambdas)
        x0_xy = np.repeat(x0_xy[np.newaxis, :], n_lam, axis=0)
        y0_xy = np.repeat(y0_xy[np.newaxis, :], n_lam, axis=0)
        lambdas = np.repeat(lambdas[:, np.newaxis], n_pixels, axis=1)
        x0s, y0s = imgxy_to_grismxy(x0_xy, y0_xy, lambdas, order)
    # x0s, y0s now have shape (n_lam, n_pixels)
    return x0s, y0s, lambdas


def disperse(
    xs,
    ys,
    source_ids_per_pixel,
    order,
    lambdas,
    direct_image_wcs,
    grism_wcs,
    naxis,
    trace_pdt=None,
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
    trace_pdt : `~jwst.wfss_contam.trace_pdt.TracePDT`, optional
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
        trace_pdt=trace_pdt,
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
