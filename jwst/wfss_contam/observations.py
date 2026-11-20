import dataclasses
import json
import logging
import multiprocessing as mp
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from photutils.background import Background2D, MedianBackground

from jwst.wfss_contam.disperse import disperse_and_model
from jwst.wfss_contam.pdt_fileio import load_pdt_from_files
from jwst.wfss_contam.transform_interp import (
    build_transform_interp,
    get_grism_detector_transform,
    native_wavelength_grid,
)

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


def _chunk_bounds_keep_sources_together(sorted_source_ids, max_pixels):
    """
    Determine chunk boundaries for a sorted list of source IDs.

    Respect a maximum pixel limit, and keep sources together within a single chunk
    unless they are larger than that limit.

    Returns
    -------
    list
        List of (start, stop) indices for each chunk.
    """
    _, source_sizes = np.unique(sorted_source_ids, return_counts=True)
    chunk_bounds = []
    batch_start = batch_stop = 0
    for source_size in source_sizes:
        source_stop = batch_stop + int(source_size)
        if source_size > max_pixels:
            while source_stop - batch_start >= max_pixels:
                chunk_stop = batch_start + max_pixels
                chunk_bounds.append((batch_start, chunk_stop))
                batch_start = chunk_stop
        elif source_stop - batch_start > max_pixels:
            chunk_bounds.append((batch_start, batch_stop))
            batch_start = batch_stop
        batch_stop = source_stop
    if batch_stop > batch_start:
        chunk_bounds.append((batch_start, batch_stop))
    return chunk_bounds


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
        segmentation_map,
        grism_wcs,
        direct_image_wcs,
        boundaries=None,
        max_cpu=1,
        max_pixels_per_chunk=5e4,
        oversample_factor=2,
        band_wavelengths=None,
        save_pdt=False,
        load_pdt=False,
        pdt_directory=None,
    ):
        """
        Initialize all data and metadata for a given observation.

        Parameters
        ----------
        direct_image : np.ndarray
            Direct imaging data.  May be 2-D ``(ny, nx)`` for a single-band
            direct image, or 3-D ``(N, ny, nx)`` for a multi-band cube.
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
        band_wavelengths : array-like of shape (N,), optional
            Central wavelengths (in microns) for each plane of a 3-D ``direct_image``.
            Required when ``direct_image`` is 3-D; ignored when ``direct_image`` is 2-D.
        save_pdt : bool
            Whether to save the PDT files to disk.
        load_pdt : bool
            Whether to reuse geometry for orders with a completed PDT manifest.
        pdt_directory : str or Path, optional
            Location of the PDT cache. Required when loading.
        """
        if load_pdt and pdt_directory is None:
            raise ValueError("pdt_directory is required when loading a PDT cache")
        self.save_pdt = save_pdt
        self.load_pdt = load_pdt
        self.pdt_directory = pdt_directory
        self._pdt_run_directory = None
        if boundaries is None:
            boundaries = []
        # Load all the info for this grism mode
        self.direct_image_wcs = direct_image_wcs
        self.grism_wcs = grism_wcs
        self.seg = segmentation_map
        self.source_ids = [sid for sid in np.unique(self.seg).tolist() if sid != 0]
        self.max_cpu = max_cpu
        self.max_pixels_per_chunk = max_pixels_per_chunk
        self.oversample_factor = oversample_factor

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
        self._create_pixel_list()

        # Initialize the output list of cutouts
        self.simulated_cutouts = []

        # Initialize the simulated dispersed image
        self.simulated_image = np.zeros(self.dims, float)

    def _create_pixel_list(self):
        """Create flat lists of pixels to be dispersed."""
        self.ys, self.xs = np.nonzero(self.seg)
        self.source_ids_per_pixel = self.seg[self.ys, self.xs]
        if self.dimage.ndim == 2:
            # Give it an extra dimension to make it shape (1, n_pixels)
            self.fluxes = self.dimage[self.ys, self.xs][np.newaxis, :]
        else:
            # Shape (N, n_pixels), where N is the number of input direct image bands
            self.fluxes = self.dimage[:, self.ys, self.xs]

    def chunk_sources(
        self,
        order,
        lambdas,
        sens_waves,
        sens_response,
        selected_ids=None,
        max_pixels=5000,
        basis_models=None,
        transform_interp=None,
    ):
        """
        Pack whole sources into bounded batches, splitting only oversized sources.

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
            Hard maximum number of input pixels per chunk. Small sources are kept
            whole; only sources whose total size exceeds this limit may be split
            to fill available batch space. Their final partial batch remains open
            for subsequent whole sources.
        transform_interp : \
            `~jwst.wfss_contam.transform_interp.DispersionTransformInterpolator`, optional
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
        if max_pixels < 1:
            raise ValueError("max_pixels must be at least 1")

        # Create a mask for selected sources
        selected_mask = np.isin(self.source_ids_per_pixel, source_ids)

        # Get pixels for selected sources
        selected_xs = self.xs[selected_mask]
        selected_ys = self.ys[selected_mask]
        selected_source_ids = self.source_ids_per_pixel[selected_mask]

        # Sort by source ID to keep sources mostly together
        sort_indices = np.argsort(selected_source_ids, kind="stable")
        sorted_xs = selected_xs[sort_indices]
        sorted_ys = selected_ys[sort_indices]
        sorted_source_ids = selected_source_ids[sort_indices]
        sorted_fluxes = self.fluxes[:, np.flatnonzero(selected_mask)[sort_indices]]

        # Build up chunks that respect the max_pixels limit while keeping sources together
        total_pixels = len(sorted_xs)
        chunk_bounds = _chunk_bounds_keep_sources_together(sorted_source_ids, max_pixels)
        log.info(
            f"Splitting {total_pixels} pixels from {len(source_ids)} sources into "
            f"{len(chunk_bounds)} chunks"
        )

        disperse_args = []
        for start_idx, end_idx in chunk_bounds:
            chunk_xs = sorted_xs[start_idx:end_idx]
            chunk_ys = sorted_ys[start_idx:end_idx]
            chunk_source_ids = sorted_source_ids[start_idx:end_idx]

            disperse_args.append(
                [
                    chunk_xs,
                    chunk_ys,
                    sorted_fluxes[:, start_idx:end_idx],
                    self.band_wavelengths,
                    chunk_source_ids,
                    order,
                    lambdas,
                    sens_waves,
                    sens_response,
                    self.direct_image_wcs,
                    self.grism_wcs,
                    self.naxis,
                    basis_models,
                    transform_interp,
                ]
            )

        return disperse_args

    def disperse_order(
        self,
        order,
        wmin,
        wmax,
        sens_waves,
        sens_response,
        selected_ids=None,
        basis_models=None,
        interp_spacing=None,
        interp_wl_oversample=1,
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
        sens_waves : ndarray
            Wavelength array from photom reference file
        sens_response : ndarray
            Response (flux calibration) array from photom reference file
        selected_ids : list, optional
            List of source IDs to process. If None, all sources are processed.
        basis_models : list of callables, optional
            Flux distributions to evaluate at each wavelength. Typically these will be single
            polynomial orders, e.g. [lambda x: x, lambda x: x^2], ...] the coefficients of which
            are linearly fit later. If None, no models are included in the output.
        interp_spacing : int, optional
            Spacing of grid points to sample along the x and y axes
            when building a cached lookup table for the trace-shape transform. If None
            (the default), no lookup table is used and the exact transform is evaluated directly
            for every pixel.
        interp_wl_oversample : float, optional
            Oversampling factor for the lookup table's wavelength grid, relative to the
            native dispersion scale. If None, the lookup table's wavelength grid instead
            exactly matches the dispersal wavelength grid, allowing
            `~jwst.wfss_contam.transform_interp.DispersionTransformInterpolator.evaluate_grid`
            to skip wavelength-axis
            interpolation entirely. Only used if ``interp_spacing`` is not None.
        """
        order_directory = (
            Path(self.pdt_directory) / f"order_{int(order)}" if self.load_pdt else None
        )
        reuse_cache = order_directory is not None and (order_directory / "manifest.json").is_file()
        save_order = self.save_pdt and not reuse_cache
        if reuse_cache:
            log.info("Reusing PDT cache in %s", order_directory)
            disperse_args = self._cached_part_args(
                order, selected_ids, sens_waves, sens_response, basis_models
            )
        else:
            disperse_args, lambdas = self._fresh_part_args(
                order,
                wmin,
                wmax,
                sens_waves,
                sens_response,
                selected_ids,
                basis_models,
                interp_spacing,
                interp_wl_oversample,
            )

        if save_order:
            directory = self._make_pdt_directory(order)
            part_counts = {}
            for args in disperse_args:
                source_parts = {}
                for source_id in np.unique(args[4]).tolist():
                    part_number = part_counts.get(source_id, 0)
                    source_parts[source_id] = part_number
                    part_counts[source_id] = part_number + 1
                args.extend((directory, source_parts))

        t0 = time.time()
        workers = min(self.max_cpu, len(disperse_args))
        log.info("Using %s CPU core(s) to model %s PDT batches...", workers, len(disperse_args))
        if workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                all_res = pool.starmap(disperse_and_model, disperse_args, chunksize=1)
        else:
            all_res = [disperse_and_model(*args) for args in disperse_args]
        log.info(f"Wall clock time for disperse_chunk order {order}: {time.time() - t0:.1f} sec")

        source_results = {}
        if save_order:
            sources = {int(sid): [] for sid in _select_ids(selected_ids, self.source_ids)}
        for results in all_res:
            if save_order:
                results, written_parts = results
                for sid, parts in written_parts.items():
                    sources[sid].extend(parts)
            for sid in results:
                _aggregate_by_source(results, sid, source_results)

        if save_order:
            temporary_manifest = directory / "manifest.tmp"
            with temporary_manifest.open("w") as manifest_file:
                json.dump({"wavelength": lambdas.tolist(), "sources": sources}, manifest_file)
            temporary_manifest.replace(directory / "manifest.json")

        self._add_source_results(order, source_results)

    def _fresh_part_args(
        self,
        order,
        wmin,
        wmax,
        sens_waves,
        sens_response,
        selected_ids,
        basis_models,
        interp_spacing,
        interp_wl_oversample,
    ):
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

        transform_interp_obj = None
        if interp_spacing is not None:
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

            if interp_wl_oversample is None:
                transform_interp_obj = build_transform_interp(
                    self.grism_wcs,
                    order,
                    wmin,
                    wmax,
                    coord_limits,
                    spacing=interp_spacing,
                    lam_grid=lambdas,
                )
            else:
                transform_interp_obj = build_transform_interp(
                    self.grism_wcs,
                    order,
                    wmin,
                    wmax,
                    coord_limits,
                    spacing=interp_spacing,
                    wave_oversample_factor=interp_wl_oversample,
                )

        # generate lists of input parameters for the disperse function
        # for each chunk of sources
        disperse_args = self.chunk_sources(
            order,
            lambdas,
            sens_waves,
            sens_response,
            selected_ids=selected_ids,
            max_pixels=self.max_pixels_per_chunk,
            basis_models=basis_models,
            transform_interp=transform_interp_obj,
        )
        return disperse_args, lambdas

    def _cached_part_args(self, order, selected_ids, sens_waves, sens_response, basis_models):
        directory = Path(self.pdt_directory) / f"order_{int(order)}"
        selected_ids = _select_ids(selected_ids, self.source_ids)
        lambdas, parts = load_pdt_from_files(directory, order, selected_ids)
        disperse_args = []
        for sid, path, input_xy in parts:
            input_x, input_y = np.asarray(input_xy, dtype=np.intp).T
            if not np.all(self.seg[input_y, input_x] == sid):
                raise ValueError(
                    f"PDT pixels do not match the current segmentation for source {sid}"
                )
            fluxes = self.dimage[..., input_y, input_x]
            if fluxes.ndim == 1:
                fluxes = fluxes[np.newaxis, :]
            disperse_args.append(
                (
                    None,
                    None,
                    fluxes,
                    self.band_wavelengths,
                    None,
                    order,
                    lambdas,
                    sens_waves,
                    sens_response,
                    None,
                    None,
                    self.naxis,
                    basis_models,
                    None,
                    None,
                    None,
                    (sid, path),
                )
            )
        return disperse_args

    def _add_source_results(self, order, source_results):
        """Add the combined source cutouts and their counts to the simulation."""
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

    def _make_pdt_directory(self, order):
        """Create the cache directory for this order."""  # numpydoc ignore: RT01
        if self._pdt_run_directory is None:
            if self.pdt_directory is None:
                self._pdt_run_directory = tempfile.mkdtemp(prefix="wfss_pdt_", dir=Path.cwd())
            else:
                requested_directory = Path(self.pdt_directory).absolute()
                directory = requested_directory.resolve()
                if requested_directory.is_symlink():
                    raise ValueError("PDT directory must not be a symbolic link")
                if (
                    directory == Path.home()
                    or directory == Path.cwd()
                    or directory in Path.cwd().parents
                ):
                    raise ValueError("PDT directory must be a dedicated output directory")
                if directory.exists() and not self.load_pdt:
                    log.warning("Overwriting existing PDT directory: %s", directory)
                    shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
                self._pdt_run_directory = str(directory)
            log.info("Writing persistent PDT files to %s", self._pdt_run_directory)
        directory = Path(self._pdt_run_directory) / f"order_{int(order)}"
        if directory.is_symlink():
            raise ValueError("PDT order directory must not be a symbolic link")
        directory.mkdir(exist_ok=True)
        if any(directory.iterdir()):
            if not self.load_pdt or (directory / "manifest.json").exists():
                raise FileExistsError(f"PDT order directory is not empty: {directory}")
            log.warning("Overwriting incomplete PDT order directory: %s", directory)
            shutil.rmtree(directory)
            directory.mkdir()
        return directory


def _aggregate_by_source(results, sid, source_results):
    """
    Combine results from different chunks into a single image and bounds for each source ID.

    Parameters
    ----------
    results : dict
        Dictionary containing the results for each source ID in the current chunk, in the format:
        {source_id: {"bounds": [xmin, xmax, ymin, ymax], "image": 2D array,
        "model_counts": list of 2D arrays}}
    sid : int
        Source ID
    source_results : dict
        Dictionary to store simulated image and bounds for each source ID,
        in the same format as results but with images and bounds aggregated.
        Updated in place.
    """
    if sid not in source_results:
        source_results[sid] = {
            "bounds": results[sid]["bounds"],
            "image": results[sid]["image"],
            "model_counts": results[sid].get("model_counts", []),
        }
        return

    # Combine bounds
    old_bounds = source_results[sid]["bounds"]
    new_bounds = results[sid]["bounds"]
    combined_bounds = [
        min(old_bounds[0], new_bounds[0]),
        max(old_bounds[1], new_bounds[1]),
        min(old_bounds[2], new_bounds[2]),
        max(old_bounds[3], new_bounds[3]),
    ]

    combined_shape = (
        combined_bounds[3] - combined_bounds[2] + 1,
        combined_bounds[1] - combined_bounds[0] + 1,
    )

    # Helper to place an existing sub-image into the combined image
    def _place(old_arr, old_b):
        out = np.zeros(combined_shape, dtype=float)
        y0 = old_b[2] - combined_bounds[2]
        x0 = old_b[0] - combined_bounds[0]
        out[y0 : y0 + old_arr.shape[0], x0 : x0 + old_arr.shape[1]] = old_arr
        return out

    old_img = _place(source_results[sid]["image"], old_bounds)
    new_img = _place(results[sid]["image"], new_bounds)
    combined_image = old_img + new_img

    # Sum model_counts across chunks (same as image)
    old_mcs = source_results[sid].get("model_counts", [])
    new_mcs = results[sid].get("model_counts", [])
    combined_mcs = [
        _place(old_mc, old_bounds) + _place(new_mc, new_bounds)
        for old_mc, new_mc in zip(old_mcs, new_mcs, strict=True)
    ]

    source_results[sid] = {
        "bounds": combined_bounds,
        "image": combined_image,
        "model_counts": combined_mcs,
    }


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
