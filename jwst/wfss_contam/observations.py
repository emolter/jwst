import dataclasses
import logging
import warnings

import numpy as np
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from photutils.background import Background2D, MedianBackground

from jwst.wfss_contam.contam_model import contam_model

log = logging.getLogger(__name__)

__all__ = ["background_subtract", "Observation", "SimulatedCutout"]


@dataclasses.dataclass
class SimulatedCutout:
    """
    Lightweight container for simulated cutout data and attributes.

    This class looks a lot like `~stdatamodels.jwst.datamodels.SlitModel` but avoids the
    schema-copy and schema-validation overhead of constructing a full SlitModel for
    every dispersed source in every spectral order. These objects are purely internal
    bookkeeping and are never added to an output datamodel.
    """

    source_id: int
    name: str
    xstart: int
    ystart: int
    xsize: int
    ysize: int
    data: np.ndarray
    spectral_order: int


def background_subtract(
    data,
    box_size=None,
    filter_size=(3, 3),
    sigma=3.0,
    exclude_percentile=30.0,
):
    """
    Apply a simple astropy background subtraction.

    Parameters
    ----------
    data : ndarray
        2D array of pixel values
    box_size : tuple
        Size of box in pixels to use for background estimation.
        If not set, defaults to 1/5 of the image size.
    filter_size : tuple
        Size of filter to use for background estimation
    sigma : float
        Sigma threshold for background clipping
    exclude_percentile : float
        Percentage of masked pixels above which box is excluded from background estimation

    Returns
    -------
    data : ndarray
        2D array of pixel values with background subtracted

    Notes
    -----
    Improper background subtraction in input _i2d image leads to extra flux
    in the simulated dispersed image, and was one cause of flux scaling issues
    in a previous version.
    """
    if box_size is None:
        box_size = (int(data.shape[0] / 5), int(data.shape[1] / 5))
    sigma_clip = SigmaClip(sigma=sigma)
    bkg_estimator = MedianBackground()
    with warnings.catch_warnings():
        # there can be multiple different AstropyUserWarning messages here about NaN and Inf values
        warnings.filterwarnings("ignore", category=AstropyUserWarning)
        bkg = Background2D(
            data,
            box_size,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            exclude_percentile=exclude_percentile,
        )
        return data - bkg.background


class Observation:
    """
    Define an observation leading to a single grism image.

    The Observation class is responsible for calling the various WCS transforms that convert
    a direct image and a segmentation image into a simulation of the grism image, making
    assumptions about the spectral properties of the direct image sources.
    When the `disperse_order` method is called one or more times, two products are created:
    the simulated dispersed image (``simulated_image`` attribute) and
    the simulated source cutouts (``simulated_cutouts`` attribute).
    """

    def __init__(
        self,
        direct_image,
        pdt,
        boundaries=None,
        band_wavelengths=None,
    ):
        """
        Initialize all data and metadata for a given observation.

        Parameters
        ----------
        direct_image : np.ndarray
            Direct imaging data.  May be 2-D ``(ny, nx)`` for a single-band
            direct image, or 3-D ``(N, ny, nx)`` for a multi-band cube.
        pdt : dict
            Table
        boundaries : list, optional
            Start/Stop coordinates of the FOV within the larger seed image.
        band_wavelengths : array-like of shape (N,), optional
            Central wavelengths (in microns) for each plane of a 3-D ``direct_image``.
            Required when ``direct_image`` is 3-D; ignored when ``direct_image`` is 2-D.
        """
        self.pdt = pdt
        if boundaries is None:
            boundaries = []
        if direct_image.ndim == 2:
            # use placeholder value since disperse() is going to see a flat SED and ignore this
            self.band_wavelengths = np.array([1.0])
            self.dimage = background_subtract(direct_image)
        else:
            # 3-D cube
            if band_wavelengths is None:
                raise ValueError("band_wavelengths must be provided when direct_image is 3-D")
            self.band_wavelengths = np.asarray(band_wavelengths, dtype=float)
            # apply background subtraction independently to each wavelength plane
            self.dimage = np.array([background_subtract(plane) for plane in direct_image])

        # Set the limits of the dispersed image to be simulated
        if len(boundaries) == 0:
            log.debug("No boundaries passed.")
            xstart = 0
            xend = xstart + self.dimage.shape[-2] - 1
            ystart = 0
            yend = ystart + self.dimage.shape[-1] - 1
        else:
            xstart, xend, ystart, yend = boundaries
        dims = (yend - ystart + 1, xend - xstart + 1)
        log.debug(f"Using simulated image size of ({dims[1]}, {dims[0]}).")

        # Initialize the output list of cutouts
        self.simulated_cutouts = []

        # Initialize the simulated dispersed image
        self.simulated_image = np.zeros(dims, float)

    def simulate_order(
        self,
        order,
        sens_waves,
        sens_response,
        basis_models=None,
    ):
        """
        Disperse the sources for a given spectral order, with multiprocessing.

        The ``simulated_cutouts`` and ``simulated_image`` attributes are updated in place.

        Parameters
        ----------
        order : int
            Spectral order to process.
        sens_waves : ndarray
            Wavelength array from photom reference file
        sens_response : ndarray
            Response (flux calibration) array from photom reference file
        basis_models : list of callables, optional
            Flux distributions to evaluate at each wavelength. Typically these will be single
            polynomial orders, e.g. [lambda x: x, lambda x: x^2], ...] the coefficients of which
            are linearly fit later. If None, no models are included in the output.
        """
        pdt_order = self.pdt[order]
        source_results = contam_model(
            pdt_order,
            self.dimage,
            self.band_wavelengths,
            sens_waves,
            sens_response,
            basis_models=basis_models,
        )

        # Now add the combined results to the simulation
        for sid in source_results:
            bounds = source_results[sid]["bounds"]
            img = source_results[sid]["image"]
            slitmodel = _construct_simulated_cutout(img, bounds, sid, order)
            fluxmodels = source_results[sid].get("model_counts", [])
            for i, fm in enumerate(fluxmodels):
                # use i+1 indexing because typically the first model will be the linear order
                # for polynomial fitting. The 0th order is what's already in slitmodel.data
                setattr(slitmodel, f"fluxmodel_{i + 1}", fm)
            self.simulated_image[bounds[2] : bounds[3] + 1, bounds[0] : bounds[1] + 1] += img
            self.simulated_cutouts.append(slitmodel)


def _construct_simulated_cutout(
    img,
    bounds,
    sid,
    order,
):
    """
    Turn an output image from a single source/order into a SimulatedCutout.

    Parameters
    ----------
    img : ndarray
        Dispersed model image of segmentation map source
    bounds : list
        The bounds of the object in relation to the full-frame image.
    sid : int
        The source ID
    order : int
        The spectral order

    Returns
    -------
    SimulatedCutout
        Simulated source cutout containing the dispersed pixel values
    """
    [thisobj_minx, thisobj_maxx, thisobj_miny, thisobj_maxy] = bounds
    return SimulatedCutout(
        source_id=sid,
        name=f"{sid}",
        # FITS pixels are 1-indexed, matching extract_2d convention
        xstart=thisobj_minx + 1,
        xsize=thisobj_maxx - thisobj_minx + 1,
        ystart=thisobj_miny + 1,
        ysize=thisobj_maxy - thisobj_miny + 1,
        # Match SlitModel float32 dtype
        data=img.astype(np.float32, copy=False),
        spectral_order=order,
    )
