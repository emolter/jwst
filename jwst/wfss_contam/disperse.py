import numpy as np
from typing import Callable, Sequence
from astropy.wcs import WCS

from scipy.interpolate import interp1d
import warnings

from ..lib.winclip import get_clipped_pixels
from .sens1d import create_1d_sens


def flux_interpolator_injector(lams: np.ndarray, 
                       flxs: np.ndarray, 
                       extrapolate_sed: bool,
                       ) -> Callable[[float], float]:
    '''
    Parameters
    ----------
    lams : float array
        Array of wavelengths corresponding to the fluxes (flxs) for each pixel.
        One wavelength per direct image, so can be a single value.
    flxs : float array
        Array of fluxes (flam) for the pixels contained in x0, y0. If a single
        direct image is in use, this will be a single value.
    extrapolate_sed : bool
        Whether to allow for the SED of the object to be extrapolated when it does not fully cover the
        needed wavelength range. Default if False.

    Returns
    -------
    flux : function
        Function that returns the flux at a given wavelength. If only one direct image is in use, this
        function will always return the same value
    '''

    if len(lams) > 1:
        # If we have direct image flux values from more than one filter (lambda),
        # we have the option to extrapolate the fluxes outside the
        # wavelength range of the direct images
        if extrapolate_sed is False:
            return interp1d(lams, flxs, fill_value=0., bounds_error=False)
        else:
            return interp1d(lams, flxs, fill_value="extrapolate", bounds_error=False)
    else:
        # If we only have flux from one lambda, just use that
        # single flux value at all wavelengths
        def flux(x):
            return flxs[0]
        return flux


def determine_wl_spacing(dw: float,
                         lams: np.ndarray, 
                         oversample_factor: int,
                         ) -> float:
    '''
    Use a natural wavelength scale or the wavelength scale of the input SED/spectrum,
    whichever is smaller, divided by oversampling requested

    Parameters
    ----------
    dw : float
        The natural wavelength scale of the grism image
    lams : float array
        Array of wavelengths corresponding to the fluxes (flxs) for each pixel.
        One wavelength per direct image, so can be a single value.
    oversample_factor : int
        The amount of oversampling

    Returns
    -------
    dlam : float
        The wavelength spacing to use for the dispersed pixels
    '''
    # 
    if len(lams) > 1:
        input_dlam = np.median(lams[1:] - lams[:-1])
        if input_dlam < dw:
            return input_dlam / oversample_factor
    return dw / oversample_factor


def dispersed_pixel(x0: np.ndarray, 
                    y0: np.ndarray,
                    width: float,
                    height: float,
                    lams: np.ndarray,
                    flxs: np.ndarray,
                    order: int,
                    wmin: float,
                    wmax: float,
                    sens_waves: np.ndarray,
                    sens_resp: np.ndarray,
                    seg_wcs: WCS, 
                    grism_wcs: WCS, 
                    ID: int,
                    naxis: Sequence[int],
                    oversample_factor: int = 2,
                    extrapolate_sed: bool = False,
                    xoffset: float = 0,
                    yoffset: float = 0,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    This function take a list of pixels and disperses them using the information contained
    in the grism image WCS object and returns a list of dispersed pixels and fluxes.

    Parameters
    ----------
    x0 : float array
        Array of x-coordinates of the centers of the pixels. One value for each
        direct image, so if only 1 direct image is in use, this is a single value.
    y0 : float array
        Array of y-coordinates of the centers of the pixels. One value for each
        direct image, so if only 1 direct image is in use, this is a single value.
    width : float
        Width of the pixels to be dispersed.
    height : float
        Height of the pixels to be dispersed.
    lams : float array
        Array of wavelengths corresponding to the fluxes (flxs) for each pixel.
        One wavelength per direct image, so can be a single value.
    flxs : float array
        Array of fluxes (flam) for the pixels contained in x0, y0. If a single
        direct image is in use, this will be a single value.
    order : int
        The spectral order to disperse.
    wmin : float
        Min wavelength to be dispersed.
    wmax : float
        Max wavelength to be dispersed.
    sens_waves : float array
        Array of wavelengths corresponding to flux calibration (sens_resp) values.
    sens_resp : float array
        Flux calibration values as a function of wavelength.
    seg_wcs : WCS object
        The WCS object of the segmentation map.
    grism_wcs : WCS object
        The WCS object of the grism image.
    ID : int
        The ID of the object to which the pixel belongs.
    naxis : tuple
        Dimensions (shape) of grism image into which pixels are dispersed.
    oversample_factor : int
        The amount of oversampling required above that of the input spectra or natural dispersion,
        whichever is smaller. Default=2.
    extrapolate_sed : bool
        Whether to allow for the SED of the object to be extrapolated when it does not fully cover the
        needed wavelength range. Default if False.
    xoffset : int
        Pixel offset to apply when computing the dispersion (accounts for offset from source cutout to
        full frame)
    yoffset : int
        Pixel offset to apply when computing the dispersion (accounts for offset from source cutout to
        full frame)

    Returns
    -------
    xs : array
        1D array of dispersed pixel x-coordinates
    ys : array
        1D array of dispersed pixel y-coordinates
    areas : array
        1D array of the areas of the incident pixel that when dispersed falls on each dispersed pixel
    lams : array
        1D array of the wavelengths of each dispersed pixel
    counts : array
        1D array of counts for each dispersed pixel
    ID : int
        The source ID. Returned for bookkeeping convenience.
    """

    # Setup the transforms we need from the input WCS objects
    sky_to_imgxy = grism_wcs.get_transform('world', 'detector')
    imgxy_to_grismxy = grism_wcs.get_transform('detector', 'grism_detector')

    # Set up function for retrieving flux values at each dispersed wavelength
    flux_interpolator = flux_interpolator_injector(lams, flxs, extrapolate_sed)

    # Get x/y positions in the grism image corresponding to wmin and wmax:
    # Start with RA/Dec of the input pixel position in segmentation map,
    # then convert to x/y in the direct image frame corresponding
    # to the grism image,
    # then finally convert to x/y in the grism image frame
    x0_sky, y0_sky = seg_wcs(x0, y0)
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, order)
    xwmin, ywmin = imgxy_to_grismxy(x0_xy + xoffset, y0_xy + yoffset, wmin, order)
    xwmax, ywmax = imgxy_to_grismxy(x0_xy + xoffset, y0_xy + yoffset, wmax, order)
    dxw = xwmax - xwmin
    dyw = ywmax - ywmin

    # Create list of wavelengths on which to compute dispersed pixels
    dw = np.abs((wmax - wmin) / (dyw - dxw))
    dlam = determine_wl_spacing(dw, lams, oversample_factor)
    lambdas = np.arange(wmin, wmax + dlam, dlam)
    n_lam = len(lambdas)

    # Compute lists of x/y positions in the grism image for
    # the set of desired wavelengths:
    # As above, first get RA/Dec of segmentation map pixel positions,
    # then convert to x/y in image frame of grism image,
    # then convert to x/y in grism frame.
    x0_sky, y0_sky = seg_wcs([x0] * n_lam, [y0] * n_lam)
    x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, lambdas, [order] * n_lam)
    x0s, y0s = imgxy_to_grismxy(x0_xy + xoffset, y0_xy + yoffset, lambdas, [order] * n_lam)

    # If none of the dispersed pixel indexes are within the image frame,
    # return a null result without wasting time doing other computations
    if x0s.min() >= naxis[0] or x0s.max() < 0 or y0s.min() >= naxis[1] or y0s.max() < 0:
        return None

    # Compute arrays of dispersed pixel locations and areas
    padding = 1
    xs, ys, areas, index = get_clipped_pixels(
        x0s, y0s,
        padding,
        naxis[0], naxis[1],
        width, height
    )
    lams = np.take(lambdas, index)

    # If results give no dispersed pixels, return null result
    if xs.size <= 1:
        return None

    # compute 1D sensitivity array corresponding to list of wavelengths
    sens, no_cal = create_1d_sens(lams, sens_waves, sens_resp)

    # Compute countrates for dispersed pixels. Note that dispersed pixel
    # values are naturally in units of physical fluxes, so we divide out
    # the sensitivity (flux calibration) values to convert to units of
    # countrate (DN/s).
    # flux_interpolator(lams) is either single-valued (for a single direct image) 
    # or an array of the same length as lams (for multiple direct images in different filters)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
        counts = flux_interpolator(lams) * areas / (sens * oversample_factor)
    counts[no_cal] = 0.  # set to zero where no flux cal info available

    return xs, ys, areas, lams, counts, ID
