import logging
import multiprocessing
import time

import numpy as np
from stcal.multiprocessing import compute_num_cores
from stdatamodels.jwst import datamodels

from jwst.assign_wcs.util import (
    get_bounding_box_extents,
    validate_orders_against_reference,
    validate_orders_against_transform,
)
from jwst.lib.catalog_utils import read_source_catalog
from jwst.wfss_contam.sens1d import get_photom_data
from jwst.wfss_geom.disperse import disperse
from jwst.wfss_geom.trace_pdt import (
    build_trace_pdt,
    get_grism_detector_transform,
    native_wavelength_grid,
)

log = logging.getLogger(__name__)


def _find_min_relresp(sens_waves, sens_response):
    """
    Find the minimum relative response in the sensitivity response.

    Helper function is necessary instead of just nanmin because reference file
    sometimes has zero-valued wavelength/response pairs.

    Parameters
    ----------
    sens_waves : np.ndarray
        Wavelengths corresponding to the sensitivity response.
    sens_response : np.ndarray
        Sensitivity response values.

    Returns
    -------
    float
        Minimum relative response value.
    """
    good = (sens_waves > 0) & (sens_response > 0) & np.isfinite(sens_waves)
    return np.nanmin(sens_response[good])


def _select_ids(source_id, all_ids):
    """
    Select the source IDs to be processed based on the input ID parameter.

    Parameters
    ----------
    source_id : int or list-like
        ID(s) of source to process. If None, all sources processed.
    all_ids : ndarray
        Array of all source IDs in the segmentation map

    Returns
    -------
    selected_IDs : list
        List of selected source IDs
    """
    if source_id is None:
        log.info(f"Loading all {len(all_ids)} sources from segmentation map")
        return all_ids

    elif isinstance(source_id, int):
        log.info(f"Loading single source {source_id} from segmentation map")
        return [source_id]

    elif isinstance(source_id, list) or isinstance(source_id, np.ndarray):
        log.info(
            f"Loading {len(source_id)} of {len(all_ids)} selected sources from segmentation map"
        )
        return list(source_id)
    else:
        raise ValueError("ID must be an integer or a list of integers")


def _apply_magnitude_limit(
    order, source_catalog, sens_wave, sens_response, magnitude_limit, min_relresp_order1
):
    """
    Rescale the magnitude limit based on the sensitivity response for a given spectral order.

    Parameters
    ----------
    order : int
        Spectral order for which the magnitude limit is applied.
    source_catalog : astropy.table.Table
        The source catalog containing source IDs and isophotal AB magnitudes.
    sens_wave : np.ndarray
        The wavelengths corresponding to the sensitivity response.
    sens_response : np.ndarray
        The sensitivity response for the order.
    magnitude_limit : float
        The isophotal AB magnitude limit for sources to be included in the contamination correction.
    min_relresp_order1 : float
        Minimum relative response for order 1, used to scale the magnitude limit.

    Returns
    -------
    list
        List of source IDs that meet the magnitude limit criteria.
    """
    if order in [0, 1]:
        # Magnitude limit is set according to order 1 sensitivity response
        # and order 0 is a special case because it's not dispersed
        order_mag_limit = magnitude_limit
    else:
        # Scale the magnitude limit according to the order sensitivity response
        order_sens_factor = min_relresp_order1 / _find_min_relresp(sens_wave, sens_response)
        order_mag_diff = -2.5 * np.log10(order_sens_factor)
        order_mag_limit = magnitude_limit - order_mag_diff

    # Select sources that are brighter than the magnitude limit
    good_sources = source_catalog[source_catalog["isophotal_abmag"] < order_mag_limit]
    if len(good_sources) == 0:
        return None
    log.info(
        f"Applying magnitude limit of {order_mag_limit:.1f} to order {order}. "
        f"Sources selected: {len(good_sources)}"
    )
    return good_sources["label"].tolist()


def _reject_off_detector_bounds(
    source_catalog, sky_to_grism, wlmin, wlmax, order, subarray_xsize, subarray_ysize
):
    """
    Determine which sources have any part of their grism-frame bounding boxes on the detector.

    Similar to the way bounding boxes are created in extract2d, but
    this version does not create GrismObjects, does not distinguish
    partially-on-detector sources, and just returns a boolean array indicating which
    sources are on the detector.

    Parameters
    ----------
    source_catalog : `~astropy.table.QTable`
        The catalog of sources with sky bounding boxes.
    sky_to_grism : callable
        A function that maps sky coordinates and wavelength to grism-frame coordinates.
        This should be the "world" to "grism_detector" transform. For MultiSlitModel remember
        that each SlitModel's input_model also has a "grism_slit" frame that must be bypassed.
    wlmin, wlmax : float
        The minimum/maximum wavelength for the grism order as set by the wavelengthrange file.
    order : int
        The spectral order to consider.
    subarray_xsize, subarray_ysize : int
        The size of the detector subarray in the x- and y-direction.

    Returns
    -------
    np.ndarray
        Boolean array indicating which sources are on the detector.
    """
    ra_corners = np.array(
        [
            source_catalog["sky_bbox_ll"].ra.value,
            source_catalog["sky_bbox_lr"].ra.value,
            source_catalog["sky_bbox_ul"].ra.value,
            source_catalog["sky_bbox_ur"].ra.value,
        ]
    )
    dec_corners = np.array(
        [
            source_catalog["sky_bbox_ll"].dec.value,
            source_catalog["sky_bbox_lr"].dec.value,
            source_catalog["sky_bbox_ul"].dec.value,
            source_catalog["sky_bbox_ur"].dec.value,
        ]
    )
    xmin, xmax, ymin, ymax = get_bounding_box_extents(
        ra_corners, dec_corners, sky_to_grism, wlmin, wlmax, order
    )
    infinite = ~np.isfinite(np.asarray([xmin, xmax, ymin, ymax]))
    any_non_finite = np.sum(infinite, axis=0) > 0

    # check against bounds
    xmin_too_big = xmin > subarray_xsize - 1
    xmax_too_small = xmax < 0
    ymin_too_big = ymin > subarray_ysize - 1
    ymax_too_small = ymax < 0
    bad = xmin_too_big | xmax_too_small | ymin_too_big | ymax_too_small | any_non_finite
    return ~bad


def _combine_geom_chunks(chunk_results, wavelength):
    """
    Combine geometry tables from individual chunks into a single order-level PDT.

    Parameters
    ----------
    chunk_results : list of dict
        List of geometry tables from individual chunks.
    wavelength : ndarray
        Array of wavelengths corresponding to the geometry tables.

    Returns
    -------
    dict
        Combined geometry table for the entire order.
    """
    input_xy = []
    source_ids = []
    grism_xy = []
    wavelength_indices = []
    fractional_areas = []
    pixel_idx_ptr = [np.array([0], dtype=np.intp)]
    n_contributions = 0

    for geometry in chunk_results:
        if geometry is None:
            continue

        input_xy.append(geometry["input_xy"])
        source_ids.append(geometry["source_id"])
        grism_xy.append(geometry["grism_xy"])
        wavelength_indices.append(geometry["wavelength_index"])
        fractional_areas.append(geometry["fractional_area"])
        pixel_idx_ptr.append(geometry["pixel_idx_ptr"][1:] + n_contributions)
        n_contributions += geometry["pixel_idx_ptr"][-1]

    return {
        "wavelength": np.asarray(wavelength),
        "input_xy": np.concatenate(input_xy) if input_xy else np.empty((0, 2), dtype=int),
        "source_id": np.concatenate(source_ids) if source_ids else np.empty(0, dtype=int),
        "pixel_idx_ptr": np.concatenate(pixel_idx_ptr),
        "grism_xy": np.concatenate(grism_xy) if grism_xy else np.empty((0, 2), dtype=int),
        "wavelength_index": (
            np.concatenate(wavelength_indices) if wavelength_indices else np.empty(0, dtype=int)
        ),
        "fractional_area": (
            np.concatenate(fractional_areas) if fractional_areas else np.empty(0, dtype=float)
        ),
    }


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
        segmentation_map,
        grism_wcs,
        direct_image_wcs,
        boundaries=None,
        max_cpu=1,
        max_pixels_per_chunk=5e4,
        oversample_factor=2,
    ):
        """
        Initialize all data and metadata for a given observation.

        Parameters
        ----------
        segmentation_map : np.ndarray
            Segmentation map data.
        grism_wcs : `~gwcs.wcs.WCS`
            WCS object from grism image
        direct_image_wcs : `~gwcs.wcs.WCS`
            WCS object from direct image
        boundaries : list, optional
            Start/Stop coordinates of the FOV within the larger seed image.
        max_cpu : int, optional
            Max number of cpu's to use when multiprocessing
        max_pixels_per_chunk : int, optional
            Maximum number of pixels per chunk when dispersing sources
        oversample_factor : int, optional
            Factor by which to oversample the wavelength grid
        """
        if boundaries is None:
            boundaries = []
        # Load all the info for this grism mode
        self.direct_image_wcs = direct_image_wcs
        self.grism_wcs = grism_wcs
        self.seg = segmentation_map
        all_ids = list(set(np.ravel(self.seg)))
        all_ids.remove(0)  # Remove the background ID
        self.source_ids = all_ids
        self.max_cpu = max_cpu
        self.max_pixels_per_chunk = max_pixels_per_chunk
        self.oversample_factor = oversample_factor

        # Set the limits of the dispersed image to be simulated
        if len(boundaries) == 0:
            log.debug("No boundaries passed.")
            self.xstart = 0
            self.xend = self.xstart + self.seg.shape[0] - 1
            self.ystart = 0
            self.yend = self.ystart + self.seg.shape[1] - 1
        else:
            self.xstart, self.xend, self.ystart, self.yend = boundaries
        self.dims = (self.yend - self.ystart + 1, self.xend - self.xstart + 1)
        log.debug(f"Using simulated image size of ({self.dims[1]}, {self.dims[0]}).")
        self.naxis = self.dims[::-1]

        # Create lists of pixels labeled in segmentation map
        self.ys, self.xs = np.nonzero(self.seg)
        self.source_ids_per_pixel = self.seg[self.ys, self.xs]

    def chunk_sources(
        self,
        order,
        lambdas,
        selected_ids=None,
        max_pixels=1e5,
        trace_pdt=None,
    ):
        """
        Chunk the sources into groups of max_pixels.

        Parameters
        ----------
        order : int
            Spectral order to process
        lambdas : ndarray
            Wavelengths at which to compute dispersed pixel values for this order,
            precomputed once per order.
        selected_ids : list, optional
            List of source IDs to process. If None, all sources are processed.
        max_pixels : int, optional
            Maximum number of pixels per chunk.
        trace_pdt : `~jwst.wfss_contam.trace_pdt.TracePDT`, optional
            Precomputed pixel dispersion table for this spectral order.
            If None, no lookup table is used.

        Returns
        -------
        disperse_args : list[list]
            Outer list has length number of groups, and each inner list contains
            the arguments to disperse() for that group
            in the format that multiprocessing starmap expects.
        """
        source_ids = _select_ids(selected_ids, self.source_ids)
        max_pixels = int(max_pixels)

        # Create a mask for selected sources
        selected_mask = np.isin(self.source_ids_per_pixel, source_ids)

        # Get pixels for selected sources
        selected_xs = self.xs[selected_mask]
        selected_ys = self.ys[selected_mask]
        selected_source_ids = self.source_ids_per_pixel[selected_mask]

        # Sort by source ID to keep sources mostly together
        # This reduces the number of times we have to call build_dispersed_image_of_source
        # within disperse()
        sort_indices = np.argsort(selected_source_ids)
        sorted_xs = selected_xs[sort_indices]
        sorted_ys = selected_ys[sort_indices]
        sorted_source_ids = selected_source_ids[sort_indices]

        # Split into chunks of max_pixels
        total_pixels = len(sorted_xs)
        n_chunks = int(np.ceil(total_pixels / max_pixels))

        log.info(
            f"Splitting {total_pixels} pixels from {len(source_ids)} sources into {n_chunks} chunks"
        )

        disperse_args = []
        for i in range(n_chunks):
            start_idx = i * max_pixels
            end_idx = min((i + 1) * max_pixels, total_pixels)

            chunk_xs = sorted_xs[start_idx:end_idx]
            chunk_ys = sorted_ys[start_idx:end_idx]
            chunk_source_ids = sorted_source_ids[start_idx:end_idx]

            disperse_args.append(
                [
                    chunk_xs,
                    chunk_ys,
                    chunk_source_ids,
                    order,
                    lambdas,
                    self.direct_image_wcs,
                    self.grism_wcs,
                    self.naxis,
                    trace_pdt,
                ]
            )

        return disperse_args

    def disperse_order(
        self,
        order,
        wmin,
        wmax,
        selected_ids=None,
        pdt_spacing=None,
        pdt_wl_oversample=1,
    ):
        """
        Disperse the sources for a given spectral order, with multiprocessing.

        The ``simulated_cutouts`` and ``simulated_image`` attributes are updated in place.

        Parameters
        ----------
        order : int
            Spectral order to process
        wmin : float
            Minimum wavelength for dispersed spectra
        wmax : float
            Maximum wavelength for dispersed spectra
        selected_ids : list, optional
            List of source IDs to process. If None, all sources are processed.
        pdt_spacing : int, optional
            Spacing of grid points to sample along the x and y axes
            when building a cached lookup table for the trace-shape transform. If None
            (the default), no lookup table is used and the exact transform is evaluated directly
            for every pixel.
        pdt_wl_oversample : float, optional
            Oversampling factor for the lookup table's wavelength grid, relative to the
            native dispersion scale. If None, the lookup table's wavelength grid instead
            exactly matches the dispersal wavelength grid, allowing
            `~jwst.wfss_contam.trace_pdt.TracePDT.evaluate_grid` to skip wavelength-axis
            interpolation entirely. Only used if ``pdt_spacing`` is not None.

        Returns
        -------
        dict
            Combined geometry table for the given spectral order.
        """
        # Determine the wavelength grid once per order.
        # native spacing does not vary significantly across the detector, so a single
        # representative pixel is good enough.
        imgxy_to_grismxy = get_grism_detector_transform(self.grism_wcs)
        nx, ny = self.naxis
        lambdas = native_wavelength_grid(
            imgxy_to_grismxy,
            order,
            wmin,
            wmax,
            (nx - 1) / 2.0,
            (ny - 1) / 2.0,
            oversample_factor=self.oversample_factor,
        )

        trace_pdt_obj = None
        if pdt_spacing is not None:
            # Determine the spatial bounding box of sources in the WFSS un-dispersed detector frame.
            # Transform the 4 corners of the direct image / segmentation map into that frame
            ny_seg, nx_seg = self.seg.shape
            corners_x = np.array([0.0, nx_seg - 1.0, nx_seg - 1.0, 0.0])
            corners_y = np.array([0.0, 0.0, ny_seg - 1.0, ny_seg - 1.0])
            x0_sky, y0_sky = self.direct_image_wcs(corners_x, corners_y, with_bounding_box=False)
            sky_to_imgxy = self.grism_wcs.get_transform("world", "detector")
            x0_xy, y0_xy, _, _ = sky_to_imgxy(x0_sky, y0_sky, 1, order)
            coord_limits = (
                min(0.0, float(np.floor(np.min(x0_xy)))),
                max(float(self.naxis[0] - 1), float(np.ceil(np.max(x0_xy)))),
                min(0.0, float(np.floor(np.min(y0_xy)))),
                max(float(self.naxis[1] - 1), float(np.ceil(np.max(y0_xy)))),
            )

            if pdt_wl_oversample is None:
                trace_pdt_obj = build_trace_pdt(
                    self.grism_wcs,
                    order,
                    wmin,
                    wmax,
                    coord_limits,
                    spacing=pdt_spacing,
                    lam_grid=lambdas,
                )
            else:
                trace_pdt_obj = build_trace_pdt(
                    self.grism_wcs,
                    order,
                    wmin,
                    wmax,
                    coord_limits,
                    spacing=pdt_spacing,
                    wave_oversample_factor=pdt_wl_oversample,
                )

        # generate lists of input parameters for the disperse function
        # for each chunk of sources
        disperse_args = self.chunk_sources(
            order,
            lambdas,
            selected_ids=selected_ids,
            max_pixels=self.max_pixels_per_chunk,
            trace_pdt=trace_pdt_obj,
        )
        t0 = time.time()
        if self.max_cpu > 1:
            # Use multiprocessing to disperse the sources
            log.info(
                f"Using {self.max_cpu} CPU cores for multiprocessing "
                f"{len(self.source_ids)} sources in {len(disperse_args)} chunks."
            )
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(self.max_cpu)
            try:
                all_res = pool.starmap(disperse, disperse_args)
            except Exception as e:
                log.error(f"Error during parallel processing: {e}")
                raise
            finally:
                pool.close()
                pool.join()
        else:
            all_res = [disperse(*args) for args in disperse_args]
        t1 = time.time()
        log.info(f"Wall clock time for disperse_chunk order {order}: {(t1 - t0):.1f} sec")

        return _combine_geom_chunks(all_res, lambdas)


def disperse_all(
    input_model,
    waverange,
    photom,
    max_cores,
    orders=None,
    magnitude_limit=None,
    max_pixels_per_chunk=5e4,
    oversample_factor=2,
    pdt_spacing=None,
    pdt_wl_oversample=1,
):
    """
    Disperse selected direct-image pixels for each order.

    Parameters
    ----------
    input_model : `~stdatamodels.jwst.datamodels.MultiSlitModel`
        Input data model containing 2D spectral cutouts. May be modified by processing:
        make a copy before calling this function, if needed.
    waverange : `~stdatamodels.jwst.datamodels.WavelengthrangeModel`
        Wavelength range reference file model
    photom : `~stdatamodels.jwst.datamodels.NrcWfssPhotomModel`, \
             `~stdatamodels.jwst.datamodels.NisWfssPhotomModel`, \
             or `~stdatamodels.jwst.datamodels.MirWfssPhotomModel`
        Photom (flux cal) reference file model
    max_cores : str or int
        Number of cores to use for multiprocessing. If set to 'none'
        (the default), then no multiprocessing will be done. The other
        allowable string values are 'quarter', 'half', and 'all', which indicate
        the fraction of cores to use for multi-proc. The total number of
        cores includes the SMT cores (Hyper Threading for Intel).
        If an integer is provided, it will be the exact number of cores used.
    orders : list, optional
        List of spectral orders to process.
        If None, all orders defined in the wavelengthrange file will be processed.
    magnitude_limit : float, optional
        Isophotal AB magnitude limit for sources to be included in the contamination correction.
        The magnitude limit is applied per spectral order, where the orders are scaled relative
        to order 0 based on their photometric response as read from the photom reference file.
        This means that generally fewer sources will be dispersed in higher orders.
        If None, no magnitude limit is applied and all sources are included.
    max_pixels_per_chunk : int, optional
        Maximum number of pixels to disperse simultaneously.
    oversample_factor : int, optional
        Wavelength oversampling factor.
    pdt_spacing : int, optional
        Pixel spacing along each of the x, y, and wavelength axes when
        building the PDT. If None, the exact grism-to-detector transform
        will be evaluated at every pixel.
    pdt_wl_oversample : float, optional
        Oversampling factor for the pixel dispersion table wavelength grid, relative to the
        native dispersion scale. If None, the lookup table's wavelength grid
        exactly matches the dispersal wavelength grid, allowing wavelength-axis
        interpolation to be skipped entirely when evaluating the PDT. Only
        used if ``pdt_spacing`` is not None.

    Returns
    -------
    input_model : `~stdatamodels.jwst.datamodels.DataModel`
        The input data model, potentially updated with contamination information.
    geometry : dict or None
        The pixel dispersion table geometry, or None if no valid spectral orders were found.
    """
    max_available_cores = multiprocessing.cpu_count()
    # don't worry about case where nchunks < ncpus; just set nchunks large for now
    ncpus = compute_num_cores(max_cores, 1e10, max_available_cores)

    # Get the direct image WCS
    direct_file = input_model.meta.direct_image
    log.debug(f"Direct image ={direct_file}")
    with datamodels.open(direct_file) as direct_model:
        direct_image_wcs = direct_model.meta.wcs

    # Get the grism WCS object from the first cutout in the input model.
    # This WCS is used to transform from direct image to grism frame for all sources
    # in the segmentation map.
    # The "detector" to "grism_detector" and "world" to "detector" transforms are identical
    # for all sources, so just use the first one. The "grism_detector" to "grism_slit"
    # transform is not used by the step.
    grism_wcs = input_model.slits[0].meta.wcs

    # Find out how many spectral orders are defined based on the
    # array of order values in the Wavelengthrange ref file,
    # then constrain the orders to the user-specified ones
    spec_orders = np.asarray(waverange.order)
    spec_orders = validate_orders_against_reference(orders, spec_orders)
    spec_orders = validate_orders_against_transform(grism_wcs, spec_orders)
    if len(spec_orders) == 0:
        log.error("No valid spectral orders found. Step will be SKIPPED.")
        return input_model, None
    log.info(f"Spectral orders requested = {[int(x) for x in spec_orders]}")

    # Get the FILTER and PUPIL wheel positions, for use later
    filter_kwd = input_model.meta.instrument.filter
    pupil_kwd = input_model.meta.instrument.pupil
    # subarray is used for MIRI photom ref file selection and model size
    subarray_kwd = input_model.meta.subarray.name
    subarray_xsize = input_model.meta.subarray.xsize
    subarray_ysize = input_model.meta.subarray.ysize

    # NOTE: The NIRCam WFSS mode uses filters that are in the FILTER wheel
    # with gratings in the PUPIL wheel. NIRISS WFSS mode, however, is just
    # the opposite. It has gratings in the FILTER wheel and filters in the
    # PUPIL wheel. So when processing NIRISS grism exposures the name of
    # filter needs to come from the PUPIL keyword value.
    if input_model.meta.instrument.name == "NIRISS":
        filter_name = pupil_kwd
    else:
        filter_name = filter_kwd

    # Read the source catalog to perform magnitude-based source selection later
    # mag limit will be scaled according to order 1 sensitivity
    source_catalog = read_source_catalog(input_model.meta.source_catalog)
    if magnitude_limit is not None:
        order1_wave_response, order1_sens_response = get_photom_data(
            photom, filter_kwd, pupil_kwd, subarray_kwd, order=1
        )
        min_relresp_order1 = _find_min_relresp(order1_wave_response, order1_sens_response)

    # set up observation object to disperse
    with datamodels.open(input_model.meta.segmentation_map) as seg_model:
        obs = Observation(
            seg_model.data,
            grism_wcs,
            direct_image_wcs,
            boundaries=[0, subarray_xsize - 1, 0, subarray_ysize - 1],
            max_cpu=ncpus,
            max_pixels_per_chunk=max_pixels_per_chunk,
            oversample_factor=oversample_factor,
        )

    no_sources = True
    pdt = {"orders": {}}
    for order in spec_orders:
        # Load lists of wavelength ranges and flux cal info
        wavelength_range = waverange.get_wfss_wavelength_range(filter_name, [order])
        wmin = wavelength_range[order][0]
        wmax = wavelength_range[order][1]
        log.debug(f"wmin={wmin}, wmax={wmax} for order {order}")
        sens_waves, sens_response = get_photom_data(
            photom, filter_kwd, pupil_kwd, subarray_kwd, order
        )

        # Constrain the source IDs to those that are below the magnitude limit
        selected_ids = None
        if magnitude_limit is not None:
            good_ids = _apply_magnitude_limit(
                order,
                source_catalog,
                sens_waves,
                sens_response,
                magnitude_limit,
                min_relresp_order1,
            )
            if not good_ids:
                log.info(
                    f"No sources meet the magnitude limit of {magnitude_limit} for order {order} ."
                    "Skipping contamination correction for this order."
                )
                continue
        else:
            good_ids = source_catalog["label"].tolist()

        # further constrain by checking against grism-frame detector extent
        source_catalog.add_index("label")
        sourcecat = source_catalog.loc[good_ids]
        is_in_bounds = _reject_off_detector_bounds(
            sourcecat,
            grism_wcs.get_transform("world", "grism_detector"),
            wmin,
            wmax,
            order,
            input_model.meta.subarray.xsize,
            input_model.meta.subarray.ysize,
        )
        selected_ids = np.array(good_ids)[is_in_bounds]
        log.info(
            f"Number of selected IDs for order {order} after checking bounding box against "
            f"detector bounds: {len(selected_ids)}"
        )
        if not len(selected_ids):
            log.info(
                f"No sources meet the magnitude limit of {magnitude_limit} for order {order} "
                "and also disperse onto the detector. "
                "Skipping contamination correction for this order."
            )
            continue
        no_sources = False

        # Compute the dispersion for all sources in this order
        log.info(f"Creating full simulated grism image for order {order}")
        pdt["orders"][int(order)] = obs.disperse_order(
            order,
            wmin,
            wmax,
            selected_ids=selected_ids,
            pdt_spacing=pdt_spacing,
            pdt_wl_oversample=pdt_wl_oversample,
        )

    if no_sources:
        log.error(
            f"No sources found that met the magnitude limit {magnitude_limit}. Step will be SKIPPED"
        )
        return None

    return pdt
