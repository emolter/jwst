"""Top-level module for WFSS contamination correction."""

import logging
import types

import numpy as np
from stdatamodels.jwst import datamodels

from jwst.wfss_contam.observations import Observation
from jwst.wfss_contam.sens1d import get_photom_data
from jwst.wfss_contam.wavefit import (
    SpectralFitError,
    apply_basis_coeffs,
    fit_cutout_by_basis_images,
)

log = logging.getLogger(__name__)

__all__ = ["contam_corr"]


class UnmatchedSourceIDError(Exception):
    """Exception raised when a source ID is not found in the list of simulated cutouts."""

    pass


class _LegendreFluxModel:
    """
    Picklable callable that evaluates the k-th Legendre polynomial.

    Callables need to be picklable so that they can be used in multiprocessing contexts.
    The wavelength argument is mapped from ``[wmin, wmax]`` to ``[-1, 1]`` before
    evaluation.
    """

    def __init__(self, k, wmin, wmax):
        self.k = k
        self.wmin = wmin
        self.wmax = wmax
        # Coefficient vector with 1 at position k and 0 elsewhere.
        self._coeffs = np.zeros(k + 1)
        self._coeffs[k] = 1.0

    def __call__(self, x):
        x_norm = 2.0 * (x - self.wmin) / (self.wmax - self.wmin) - 1.0
        return np.polynomial.legendre.legval(x_norm, self._coeffs)


def _find_matching_simul_cutout(slitmodel, simul_cutout_sids, simul_cutout_orders):
    """
    Find the index of the matching simulated cutout in the list of simulated cutouts.

    Parameters
    ----------
    slitmodel : `~stdatamodels.jwst.datamodels.SlitModel`
        Source cutout model
    simul_cutout_sids : list
        List of source IDs for simulated cutouts
    simul_cutout_orders : list
        List of spectral orders for simulated cutouts

    Returns
    -------
    good_idx : int
        Index of the matching simulated source in the list of sids
    """
    sid = slitmodel.source_id
    order = slitmodel.meta.wcsinfo.spectral_order
    good = (np.array(simul_cutout_sids) == sid) * (np.array(simul_cutout_orders) == order)
    if not any(good):
        raise UnmatchedSourceIDError(
            f"Source ID {sid} order {order} requested by input model "
            "but not found in simulated cutouts. "
            "Contamination from that source will not be included."
        )
    return np.where(good)[0][0]


def _cut_frame_to_match_cutout(contam, slitmodel):
    """
    Cut out the contamination image to match the extent of the source cutout.

    Parameters
    ----------
    contam : 2D array
        Contamination image for the full grism exposure
    slitmodel : `~stdatamodels.jwst.datamodels.SlitModel`
        Source cutout model

    Returns
    -------
    cutout : 2D array
        Contamination image cutout that matches the extent of the source cutout
    """
    x1 = slitmodel.xstart - 1  # convert FITS 1-indexed to 0-indexed for array access
    y1 = slitmodel.ystart - 1  # convert FITS 1-indexed to 0-indexed for array access
    xf = x1 + slitmodel.xsize
    yf = y1 + slitmodel.ysize

    # zero-pad the contamination image if the cutout extends beyond the contamination image
    # fixes an off-by-one bug when sources extend to the edge of the contamination image
    if xf > contam.shape[1]:
        contam = np.pad(contam, ((0, 0), (0, xf - contam.shape[1])), mode="constant")
    if yf > contam.shape[0]:
        contam = np.pad(contam, ((0, yf - contam.shape[0]), (0, 0)), mode="constant")

    return contam[y1 : y1 + slitmodel.ysize, x1 : x1 + slitmodel.xsize]


class CutoutOverlapError(Exception):
    """Exception raised when there is no overlap between data and model for a cutout."""

    pass


def match_backplane_prefer_first(slit0, slit1):
    """
    Reshape slit1 to the backplane of slit0.

    Parameters
    ----------
    slit0 : `~stdatamodels.jwst.datamodels.SlitModel`
        Slit model for the first slit, which is used as reference.
    slit1 : `~stdatamodels.jwst.datamodels.SlitModel`
        Slit model for the second slit, which is reshaped to match slit0.

    Returns
    -------
    slit1 : `~stdatamodels.jwst.datamodels.SlitModel`
        Reshaped slit model.
    """
    data0 = slit0.data
    data1 = slit1.data

    x1 = slit1.xstart - slit0.xstart
    y1 = slit1.ystart - slit0.ystart
    backplane1 = np.zeros_like(data0)

    i0 = max([y1, 0])
    i1 = min([y1 + data1.shape[0], data0.shape[0], data1.shape[0]])
    j0 = max([x1, 0])
    j1 = min([x1 + data1.shape[1], data0.shape[1], data1.shape[1]])
    if i0 >= i1 or j0 >= j1:
        raise CutoutOverlapError(
            f"No overlap region between data and model for source {slit0.source_id}, "
            f"order {slit0.meta.wcsinfo.spectral_order}. "
            "Contamination from that source will not be included."
        )
    di = i0 - y1  # offset into data1's own row axis
    dj = j0 - x1  # offset into data1's own col axis
    backplane1[i0:i1, j0:j1] = data1[di : di + (i1 - i0), dj : dj + (j1 - j0)]
    slit1.data = backplane1

    # also update fluxmodel attributes if present
    k = 1
    while True:
        attr = f"fluxmodel_{k}"
        mc = getattr(slit1, attr, None)
        if mc is None:
            break
        mc_bp = np.zeros_like(data0)
        mc_bp[i0:i1, j0:j1] = np.asarray(mc)[di : di + (i1 - i0), dj : dj + (j1 - j0)]
        setattr(slit1, attr, mc_bp)
        k += 1

    slit1.xstart = slit0.xstart
    slit1.ystart = slit0.ystart
    slit1.xsize = slit0.xsize
    slit1.ysize = slit0.ysize

    return slit1


def _build_simulated_image_from_cutouts(simulated_cutouts, shape):
    """
    Reconstruct the full-frame simulated image from the simulated cutouts.

    This is needed instead of just using ``obs.simulated_image`` because when
    spectral fitting is requested, the simulated cutouts get modified, and the
    modifications need to end up in the simul file.

    Parameters
    ----------
    simulated_cutouts : list of `~stdatamodels.jwst.datamodels.SlitModel`
        The simulated source cutouts.
    shape : tuple of int
        ``(nrows, ncols)`` of the full detector frame.

    Returns
    -------
    full_image : ndarray
        Full-frame simulated image.
    """
    full_image = np.zeros(shape, dtype=float)
    nrows, ncols = shape
    for slitmodel in simulated_cutouts:
        x0 = slitmodel.xstart - 1  # convert FITS 1-indexed to 0-indexed for array access
        y0 = slitmodel.ystart - 1  # convert FITS 1-indexed to 0-indexed for array access
        # Clip to frame boundaries in case a cutout overflows
        x1 = min(x0 + slitmodel.xsize, ncols)
        y1 = min(y0 + slitmodel.ysize, nrows)
        full_image[y0:y1, x0:x1] += slitmodel.data[: y1 - y0, : x1 - x0]
    return full_image


def _match_simulated_cutouts(output_model, obs):
    """
    Match each observed source to its simulated counterpart.

    Reprojects the simulated cutout onto the same backplane as the observed cutout.
    If the source ID is not found in the simulated cutouts,
    or if there is no spatial overlap between the observed cutout and the simulated cutout,
    the cutout is skipped and represented as ``None`` in the returned list.

    Parameters
    ----------
    output_model : `~stdatamodels.jwst.datamodels.MultiSlitModel`
        The observed cutouts to match against.
    obs : `~jwst.wfss_contam.observations.Observation`
        The observation object containing the dispersed simulated sources.

    Returns
    -------
    matched_flat_simuls : list of `~stdatamodels.jwst.datamodels.SlitModel`
        Simulated cutout reprojected onto each observed cutout's backplane,
        or ``None`` where no match was found.
    good_idxs : list of int or None
        Index into ``obs.simulated_cutouts`` for each observed cutout,
        or ``None`` where no match was found.
    """
    simul_sids = [s.source_id for s in obs.simulated_cutouts]
    simul_orders = [s.spectral_order for s in obs.simulated_cutouts]

    matched_flat_simuls = []
    good_idxs = []
    for cutout in output_model.slits:
        try:
            good_idx = _find_matching_simul_cutout(cutout, simul_sids, simul_orders)
            # Copy only the data arrays and metadata specifically needed by
            # match_backplane_prefer_first and fit_cutout_by_basis_images.
            # This reduces overall peak memory usage of the step by a factor of ~3 in some cases
            src = obs.simulated_cutouts[good_idx]
            matched_flat = types.SimpleNamespace(
                data=np.array(src.data),
                xstart=src.xstart,
                ystart=src.ystart,
                xsize=src.xsize,
                ysize=src.ysize,
            )
            k = 1
            while True:
                mc = getattr(src, f"fluxmodel_{k}", None)
                if mc is None:
                    break
                setattr(matched_flat, f"fluxmodel_{k}", np.array(mc))
                k += 1
            matched_flat = match_backplane_prefer_first(cutout, matched_flat)
            matched_flat_simuls.append(matched_flat)
            good_idxs.append(good_idx)
        except (UnmatchedSourceIDError, CutoutOverlapError) as e:
            log.warning(e)
            matched_flat_simuls.append(None)
            good_idxs.append(None)

    return matched_flat_simuls, good_idxs


def _fit_spectral_shape(
    observed_cutout,
    simul_cutout,
    simul_cutout_backplane_unmatched,
    polyfit_degree,
    l2_alpha=0.0,
    rejection_threshold=0.1,
):
    """
    Fit a polynomial spectral shape to one cutout and apply the result in-place.

    Parameters
    ----------
    observed_cutout : `~stdatamodels.jwst.datamodels.SlitModel`
        Observed cutout whose ``.contamcorr_data`` is used as the target for the fit.
        If ``n_iterations > 1``, this holds the contamination-corrected data
        from the previous iteration. The original ``.data`` is left unchanged.
    simul_cutout : `~stdatamodels.jwst.datamodels.SlitModel`
        Simulated cutout, already backplane-matched to ``observed_cutout``.
        Its ``.data`` attribute is updated in-place with the spectrally fitted result.
    simul_cutout_backplane_unmatched : `~stdatamodels.jwst.datamodels.SlitModel`
        The simulation in ``obs.simulated_cutouts`` without backplane matching applied.
        This is tracked independently from ``simul_cutout`` because the extraction
        of the observed source cutout from extract_2d often misses flux from the extended PSF wings,
        whereas the simulation in this step disperses the whole segment, including those wings.
        Its ``.data`` is updated in-place so the full-frame reconstruction
        reflects the fitted spectral shape.
    polyfit_degree : int or None
        Degree of the polynomial spectral model.  ``None`` means no fitting.
    l2_alpha : float, optional
        L2 regularisation strength passed to `~jwst.wfss_contam.wavefit.fit_cutout_by_basis_images`.
    rejection_threshold : float, optional
        Threshold for rejecting fits based on the fitted constant term coefficient, passed to
        `~jwst.wfss_contam.wavefit.fit_cutout_by_basis_images`.

    Returns
    -------
    status : bool
        ``True`` if the fit was successful, ``False`` if the fit failed and the
        original flat-spectrum simulation is used.
    """
    if polyfit_degree is None or getattr(simul_cutout, "fluxmodel_1", None) is None:
        return False

    log.debug(
        f"Fitting polynomial of degree {polyfit_degree} to the simulation "
        f"for source ID {observed_cutout.source_id}, "
        f"order {observed_cutout.meta.wcsinfo.spectral_order}"
    )
    try:
        coeffs = fit_cutout_by_basis_images(
            observed_cutout,
            simul_cutout,
            l2_alpha=l2_alpha,
            rejection_threshold=rejection_threshold,
        )
        if coeffs is None:
            return False
        simul_cutout.data = apply_basis_coeffs(simul_cutout, coeffs)
        simul_cutout_backplane_unmatched.data = apply_basis_coeffs(
            simul_cutout_backplane_unmatched, coeffs
        )
    except SpectralFitError as e:
        log.debug(
            f"Polynomial fitting failed for cutout with source ID {observed_cutout.source_id}, "
            f"order {observed_cutout.meta.wcsinfo.spectral_order}: {e}. "
            "Using the original simulated cutout without fitting."
        )
        return False
    else:
        return True


def _build_contam(output_model, per_source_simuls, simul_data, original_data):
    """
    Build the contamination model for each source.

    Parameters
    ----------
    output_model : `~stdatamodels.jwst.datamodels.MultiSlitModel`
        The output model containing the observed spectral cutouts.
    per_source_simuls : list
        List of simulated spectra corresponding to each observed cutout.
    simul_data : np.ndarray
        The full-frame simulated data.
    original_data : list
        List of original observed data arrays, retained as the baseline for
        computing ``cutout.contamcorr_data``.

    Returns
    -------
    list
        List of contamination cutouts for each source.
    """
    contam_cuts = []
    for i, (cutout, this_simul) in enumerate(
        zip(output_model.slits, per_source_simuls, strict=True)
    ):
        if this_simul is None:
            contam_cut = np.zeros_like(cutout.data)
        else:
            simul_all_cut = _cut_frame_to_match_cutout(simul_data, cutout)
            contam_cut = simul_all_cut - this_simul.data
        cutout.contamcorr_data = original_data[i] - contam_cut
        contam_cuts.append(contam_cut)
    return contam_cuts


def contam_corr(
    input_model,
    pdt,
    waverange,
    photom,
    polyfit_degree=None,
    n_iterations=1,
    l2_alpha=0.1,
    rejection_threshold=0.1,
):
    """
    Correct contamination in WFSS spectral cutouts.

    Parameters
    ----------
    input_model : `~stdatamodels.jwst.datamodels.MultiSlitModel`
        Input data model containing 2D spectral cutouts. May be modified by processing:
        make a copy before calling this function, if needed.
    pdt : dict
        Dictionary of precomputed geometry for each spectral order.
    waverange : `~stdatamodels.jwst.datamodels.WavelengthrangeModel`
        Wavelength range reference file model
    photom : `~stdatamodels.jwst.datamodels.NrcWfssPhotomModel`, \
             `~stdatamodels.jwst.datamodels.NisWfssPhotomModel`, \
             or `~stdatamodels.jwst.datamodels.MirWfssPhotomModel`
        Photom (flux cal) reference file model
    polyfit_degree : int, optional
        Degree of polynomial fit to spectral shape. If None (the default), do not attempt
        polynomial fitting and just use the flat-spectrum simulation.
    n_iterations : int, optional
        Number of times to iterate the contamination correction. On each iteration the
        polynomial fit is re-run using the contamination-corrected spectrum from the
        previous iteration, yielding a progressively better estimate of each source's
        true spectral shape (and therefore a better contamination estimate for its
        neighbors). Requires ``polyfit_degree`` to be set; if ``polyfit_degree`` is
        None this parameter is ignored and a single iteration is performed.
    l2_alpha : float, optional
        L2 regularization strength for the polynomial spectral fit, passed to
        `~jwst.wfss_contam.wavefit.fit_cutout_by_basis_images`.
    rejection_threshold : float, optional
        Threshold for rejecting fits based on the fitted constant term coefficient, passed to
        `~jwst.wfss_contam.wavefit.fit_cutout_by_basis_images`.

    Returns
    -------
    output_model : `~stdatamodels.jwst.datamodels.MultiSlitModel`
        A copy of the input_model with contamination-corrected data stored in
        each slit's ``contamcorr_data`` attribute. The original ``data`` arrays are unchanged.
    simul_model : `~stdatamodels.jwst.datamodels.ImageModel`
        Full-frame simulated image of the grism exposure
    """
    # Get the segmentation map and direct image for this grism exposure
    direct_file = input_model.meta.direct_image
    log.debug(f"Direct image ={direct_file}")
    with datamodels.open(direct_file) as direct_model:
        band_wavelengths = None
        if isinstance(direct_model, datamodels.WFSSCubeModel):
            # Multi-band direct image: each wavelength plane holds the flux in that band.
            band_wavelengths = direct_model.wavelength.flatten().astype(float)
            log.info(
                "Direct image is a WFSSCubeModel with "
                f"{len(band_wavelengths)} wavelength planes "
                f"covering {band_wavelengths[0]:.4f} to {band_wavelengths[-1]:.4f} microns"
            )
        direct_image = direct_model.data

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

    # set up observation object to model
    obs = Observation(
        direct_image,
        pdt,
        boundaries=[0, subarray_xsize - 1, 0, subarray_ysize - 1],
        band_wavelengths=band_wavelengths,
    )

    for order in pdt.keys():
        # Load lists of wavelength ranges and flux cal info
        wavelength_range = waverange.get_wfss_wavelength_range(filter_name, [order])
        wmin = wavelength_range[order][0]
        wmax = wavelength_range[order][1]
        log.debug(f"wmin={wmin}, wmax={wmax} for order {order}")
        sens_waves, sens_response = get_photom_data(
            photom, filter_kwd, pupil_kwd, subarray_kwd, order
        )

        # Build Legendre basis flux models for disperse() if polynomial fitting is requested.
        # wmin/wmax are order-specific, so construction must happen inside the order loop.
        # The constant term (k=0, i.e. slitmodel.data) is always included; start at degree 1.
        basis_models = None
        if polyfit_degree is not None:
            basis_models = [_LegendreFluxModel(k, wmin, wmax) for k in range(1, polyfit_degree + 1)]

        # Compute the dispersion for all sources in this order
        log.info(f"Creating full simulated grism image for order {order}")
        obs.simulate_order(
            order,
            sens_waves,
            sens_response,
            basis_models=basis_models,
        )

    # Copy the input model so that adding result attributes does not modify it,
    # then retain only cutouts with simulated counterparts.
    output_model = input_model.copy()
    for i in reversed(range(len(output_model.slits))):
        if output_model.slits[i].source_id not in [c.source_id for c in obs.simulated_cutouts]:
            del output_model.slits[i]

    # Hold onto the unchanged input data so iterative corrections always start
    # from the same baseline. Corrected data is stored separately in contamcorr_data.
    original_data = [cutout.data for cutout in output_model.slits]

    if n_iterations > 1 and polyfit_degree is None:
        log.warning(
            "n_iterations > 1 has no effect when polyfit_degree is None "
            "(there is no spectral fit to iterate). Only one iteration will be performed."
        )
        n_iterations = 1

    # Match simulated sources to observed sources
    matched_flat_simuls, good_idxs = _match_simulated_cutouts(output_model, obs)

    if polyfit_degree is not None:
        # Iterate: each pass re-fits spectral shapes using the contamination-corrected
        # data from the previous pass, giving progressively better contamination estimates.
        log.info(
            f"Using polyfit_degree={polyfit_degree} "
            f"for spectral fitting over {n_iterations} iterations"
        )
        # check that background subtraction did not fail
        if input_model.meta.cal_step.bkg_subtract != "COMPLETE":
            log.warning(
                f"Background subtraction step status is {input_model.meta.cal_step.bkg_subtract}. "
                "A good background subtraction is necessary for models to match observed data "
                "well enough for spectral fitting to succeed. Fitting will be attempted, "
                "but failures may be expected."
            )

    # Save the brightness of the simulation for each spectrum for sorting later.
    # _fit_spectral_shape only reassigns the simulated cutout's .data (never
    # modifies fluxmodel_N), so the flat data captures the intrinsic source flux
    # independently of any fitted shape.
    flat_matched_sum = [
        np.nansum(np.array(s.data)) if s is not None else None for s in matched_flat_simuls
    ]

    # Apply flat-spectrum contamination correction
    # If fitting is requested, fit will start with flat-contam-removed data
    per_source_simuls = list(matched_flat_simuls)
    simul_data = _build_simulated_image_from_cutouts(
        obs.simulated_cutouts, obs.simulated_image.shape
    )
    contam_cuts = _build_contam(output_model, per_source_simuls, simul_data, original_data)

    if polyfit_degree is not None:
        for iteration in range(n_iterations):
            log.info(f"Contamination correction iteration {iteration + 1} of {n_iterations}")

            per_source_simuls = list(matched_flat_simuls)

            # Sort fittable sources by decreasing brightness so brighter sources are fitted
            # first.  Their corrected simulations are immediately folded into simul_data
            # so subsequent fainter sources see better contamination within this iteration.
            fittable = [
                k for k in range(len(output_model.slits)) if matched_flat_simuls[k] is not None
            ]
            sort_order = sorted(fittable, key=lambda k: -flat_matched_sum[k])

            # Build simulation from the previous iteration's fitted shapes (flat spectrum for
            # the first iteration).  Update incrementally after each successful fit so that
            # fainter sources benefit from the best available contamination estimate.
            simul_data = _build_simulated_image_from_cutouts(
                obs.simulated_cutouts, obs.simulated_image.shape
            )
            success = 0
            for i in sort_order:
                cutout = output_model.slits[i]
                matched_flat = matched_flat_simuls[i]

                # Compute the latest contamination estimate
                # For sources brighter than the ith one in the sort order, this will
                # include the polyfit from this order.
                # For fainter sources, it will be whatever was in the previous iteration,
                # i.e., flat-spectrum for the first iteration.
                simul_all_cut = _cut_frame_to_match_cutout(simul_data, cutout)
                cutout.contamcorr_data = original_data[i] - (simul_all_cut - matched_flat.data)

                if _fit_spectral_shape(
                    cutout,
                    matched_flat,
                    obs.simulated_cutouts[good_idxs[i]],
                    polyfit_degree,
                    l2_alpha=l2_alpha,
                    rejection_threshold=rejection_threshold,
                ):
                    success += 1
                    # Immediately rebuild so subsequent (fainter) sources see updated fit
                    simul_data = _build_simulated_image_from_cutouts(
                        obs.simulated_cutouts, obs.simulated_image.shape
                    )

            log.info(
                f"Spectral fitting successful for {success} out of "
                f"{len(output_model.slits)} sources "
                f"in iteration {iteration + 1}. Turn on debug logging for details of failures."
            )

            # Compute per-source contamination and update corrected data for the next iteration.
            # Always subtract from the original input so errors do not accumulate across iterations.
            contam_cuts = _build_contam(output_model, per_source_simuls, simul_data, original_data)

            if success == 0:
                log.warning(
                    f"No successful spectral fits in iteration {iteration + 1}. "
                    "Will not continue iterating. Ensure that the background is well subtracted, "
                    "and consider reducing polyfit_degree or increasing l2_alpha to improve fit "
                    "stability."
                )
                break

    # Build output simulated cutouts and contam estimates from the final iteration's results.
    log.info("Creating contamination image for each individual source")
    for i in range(len(per_source_simuls)):
        this_simul = per_source_simuls[i]
        contam_cut = contam_cuts[i]
        if this_simul is not None:
            output_model.slits[i].simul = this_simul.data

        output_model.slits[i].contam = contam_cut

    simul_model = datamodels.ImageModel(data=simul_data)
    simul_model.update(input_model, only="PRIMARY")

    output_model.update(input_model, only="PRIMARY")
    output_model.meta.cal_step.wfss_contam = "COMPLETE"

    return output_model, simul_model
