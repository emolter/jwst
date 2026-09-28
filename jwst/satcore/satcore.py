import numpy as np
import stpsf
from photutils.psf import PSFPhotometry
from photutils.segmentation import SourceCatalog, SourceFinder
from scipy.ndimage import distance_transform_edt
from stdatamodels.jwst.datamodels import dqflags

from jwst.tweakreg.tweakreg_catalog import JWSTBackground


def _fill_nan_with_nearest(array):
    """
    Return a copy with non-finite pixels replaced by nearest finite values.

    Parameters
    ----------
    array : np.ndarray
        The input image data with NaNs.

    Returns
    -------
    np.ndarray
        A copy of the input array with non-finite pixels replaced by the nearest finite values.
    """
    result = np.array(array, copy=True)
    invalid = ~np.isfinite(result)
    if not np.any(invalid):
        return result
    if np.all(invalid):
        raise ValueError("Cannot fill an array with no finite pixels.")

    nearest_indices = distance_transform_edt(
        invalid,
        return_distances=False,
        return_indices=True,
    )
    result[invalid] = result[tuple(nearest_indices[:, invalid])]
    return result


def _find_bright_sources(infill_data, mask=None):  # noqa: ARG001
    """
    Detect very bright sources, including saturated sources, using segmentation.

    Inputs and outputs are designed to be compatible with the PSFPhotometry class,
    i.e., this function's signature looks like IRAFStarFinder or DAOStarFinder.
    Source finding parameters are tuned to find bright sources.
    The data are expected to have the NaNs in their saturated cores filled in before
    calling this function, and the NaN mask is intentionally not used to minimize
    issues where single stars with large saturated regions would be split into multiple sources.

    Parameters
    ----------
    infill_data : np.ndarray
        The image data with NaNs filled in using nearest finite values.
    mask : np.ndarray, optional
        Not used, but required for compatibility with the PSFPhotometry class.

    Returns
    -------
    table : `~astropy.table.Table` or None
        A table of detected source positions with columns "xcentroid" and "ycentroid".
        Returns None if no sources are detected.
    """
    detection_threshold = 10.0
    segmentation_finder = SourceFinder(
        n_pixels=7, connectivity=8, deblend=True, n_levels=8, contrast=0.5
    )
    segmentation = segmentation_finder(infill_data, detection_threshold)
    if segmentation is None:
        return None
    catalog = SourceCatalog(infill_data, segmentation)
    return catalog.to_table(columns=("x_centroid", "y_centroid"))


def _replace_cores_with_model(data, fit_image, x_fit, y_fit, box_halfwidth=10):
    """
    Replace NaN pixels near the fitted PSF positions with the corresponding model values.

    Parameters
    ----------
    data : np.ndarray
        The original image data with NaNs.
    fit_image : np.ndarray
        The model image generated from the PSF fitting.
    x_fit : array-like
        The x-coordinates of the fitted PSF positions.
    y_fit : array-like
        The y-coordinates of the fitted PSF positions.
    box_halfwidth : int, optional
        The half-width of the box around each fitted position to replace NaN pixels.

    Returns
    -------
    result : np.ndarray
        A copy of the original data with NaN pixels near source centers replaced by model values.
    full_mask : np.ndarray
        A mask indicating the pixels that were replaced, with 1 for replaced pixels and 0 otherwise.
    """
    result = np.array(data, copy=True)
    full_mask = np.zeros_like(result)
    for xc, yc in zip(x_fit, y_fit, strict=True):
        y_min = int(max(yc - box_halfwidth, 0))
        y_max = int(min(yc + box_halfwidth + 1, result.shape[0]))
        x_min = int(max(xc - box_halfwidth, 0))
        x_max = int(min(xc + box_halfwidth + 1, result.shape[1]))
        mask = ~np.isfinite(result[y_min:y_max, x_min:x_max])
        full_mask[y_min:y_max, x_min:x_max][mask] = 1
        result[y_min:y_max, x_min:x_max][mask] = fit_image[y_min:y_max, x_min:x_max][mask]
    return result, full_mask


def infill_saturated_cores(model, oversample=3, num_psfs=36, fov_pixels=51, replace_boxsize=20):
    """
    Find saturated point source in the image and fill them with simulated PSF models.

    Source finding takes place using a NaN-infilled dataset with a segmentation algorithm.
    This helps ensure large saturated regions are detected as a single source and the PSF fitter
    is able to start relatively near the true center.

    The PSFPhotometry is subject to a mask that excludes the pixels that were originally NaN.
    This excludes non-finite pixels from the fitting process.

    Parameters
    ----------
    model : `~jwst.datamodels.DataModel`
        The input data model containing the image with saturated point sources.
    oversample : int, optional
        The oversampling factor for the PSF model, by default 3.
    num_psfs : int, optional
        The number of PSFs to generate for the PSF grid, by default 36.
    fov_pixels : int, optional
        The size of the field of view in pixels for the PSF fitting, by default 51.
    replace_boxsize : int, optional
        The size of the box around each fitted position to replace NaN pixels, by default 20.

    Returns
    -------
    result : `~jwst.datamodels.DataModel`
        The output data model with saturated cores filled with simulated PSF models.
    """
    data = model.data
    bkg = JWSTBackground(data, box_size=400)
    data -= bkg.background
    mask = ~np.isfinite(data)
    err = model.err

    # Replace NaNs in input data so the source detector can find saturated cores
    infill_data = _fill_nan_with_nearest(data)

    # very helpful stpsf utility handles all instrument/filter setup for us
    inst = stpsf.setup_sim_to_match_file(model)
    grid = inst.psf_grid(
        num_psfs=num_psfs,
        all_detectors=False,
        verbose=True,
        oversample=oversample,
        fov_pixels=fov_pixels,
    )

    fit_shape = (fov_pixels, fov_pixels)
    psfphot = PSFPhotometry(
        grid, fit_shape, finder=_find_bright_sources, aperture_radius=fov_pixels // 2
    )

    # The source detector is specifically designed to ignore the mask and expect infilled data.
    # However inputting the mask here means it will get used to mask all the infilled pixels during
    # the PSF fitting portion of the algorithm. So this setup gives us the best of both worlds-
    # we effectively are using infilled data for source finding, and original data for fitting.
    results = psfphot(infill_data, mask=mask, error=err)
    fit_image = psfphot.make_model_image(data.shape)

    simcore_image, simcore_mask = _replace_cores_with_model(
        data,
        fit_image,
        results["x_fit"],
        results["y_fit"],
        box_halfwidth=replace_boxsize // 2,
    )

    # add the background back in
    simcore_image += bkg.background
    model.data = simcore_image
    dqval = dqflags.pixel["FLUX_ESTIMATED"]
    model.dq[simcore_mask.astype(bool)] = dqval
    return model
