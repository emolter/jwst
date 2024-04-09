import logging
import multiprocessing
import numpy as np

from stdatamodels.jwst import datamodels
from astropy.table import Table

from .observations import Observation
from .sens1d import get_photom_data

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _determine_multiprocessing_ncores(max_cores, num_cores):

    """Determine the number of cores to use for multiprocessing.

    Parameters
    ----------
    max_cores : string
        See docstring of contam_corr
    num_cores : int
        Number of cores available on the machine

    Returns
    -------
    ncpus : int
        Number of cores to use for multiprocessing

    """
    if max_cores == 'none':
        ncpus = 1
    else:
        if max_cores == 'quarter':
            ncpus = num_cores // 4 or 1
        elif max_cores == 'half':
            ncpus = num_cores // 2 or 1
        elif max_cores == 'all':
            ncpus = num_cores
        else:
            ncpus = 1
        log.debug(f"Found {num_cores} cores; using {ncpus}")

    return ncpus


def contam_corr(input_model, waverange, photom, max_cores, brightest_n=None):
    """
    The main WFSS contamination correction function

    Parameters
    ----------
    input_model : `~jwst.datamodels.MultiSlitModel`
        Input data model containing 2D spectral cutouts
    waverange : `~jwst.datamodels.WavelengthrangeModel`
        Wavelength range reference file model
    photom : `~jwst.datamodels.NrcWfssPhotomModel` or `~jwst.datamodels.NisWfssPhotomModel`
        Photom (flux cal) reference file model    
    max_cores : string
        Number of cores to use for multiprocessing. If set to 'none'
        (the default), then no multiprocessing will be done. The other
        allowable values are 'quarter', 'half', and 'all', which indicate
        the fraction of cores to use for multi-proc. The total number of
        cores includes the SMT cores (Hyper Threading for Intel).
    brightest_n : int
        Number of sources to simulate. If None, then all sources in the
        input model will be simulated. Requires loading the source catalog
        file if not None. Note runtime scales non-linearly with this number
        because brightest (and therefore typically largest) sources are
        simulated first.

    Returns
    -------
    output_model : `~jwst.datamodels.MultiSlitModel`
        A copy of the input_model that has been corrected
    simul_model : `~jwst.datamodels.ImageModel`
        Full-frame simulated image of the grism exposure
    contam_model : `~jwst.datamodels.MultiSlitModel`
        Contamination estimate images for each source slit

    """

    num_cores = multiprocessing.cpu_count()
    ncpus = _determine_multiprocessing_ncores(max_cores, num_cores)

    # Initialize output model
    output_model = input_model.copy()

    # Get the segmentation map, direct image for this grism exposure
    seg_model = datamodels.open(input_model.meta.segmentation_map)
    direct_file = input_model.meta.direct_image
    image_names = [direct_file]
    log.debug(f"Direct image names={image_names}")

    # Get the grism WCS object and offsets from the first cutout in the input model.
    # This WCS is used to transform from direct image to grism frame for all sources
    # in the segmentation map - the offsets are required so that we can shift
    # each source in the segmentation map to the proper grism image location
    # using this particular wcs, but any cutout's wcs+offsets would work.
    grism_wcs = input_model.slits[0].meta.wcs
    xoffset = input_model.slits[0].xstart - 1
    yoffset = input_model.slits[0].ystart - 1

    # Find out how many spectral orders are defined, based on the
    # array of order values in the Wavelengthrange ref file
    spec_orders = np.asarray(waverange.order)
    spec_orders = spec_orders[spec_orders != 0]  # ignore any order 0 entries
    log.debug(f"Spectral orders defined = {spec_orders}")

    # Get the FILTER and PUPIL wheel positions, for use later
    filter_kwd = input_model.meta.instrument.filter
    pupil_kwd = input_model.meta.instrument.pupil

    # NOTE: The NIRCam WFSS mode uses filters that are in the FILTER wheel
    # with gratings in the PUPIL wheel. NIRISS WFSS mode, however, is just
    # the opposite. It has gratings in the FILTER wheel and filters in the
    # PUPIL wheel. So when processing NIRISS grism exposures the name of
    # filter needs to come from the PUPIL keyword value.
    if input_model.meta.instrument.name == 'NIRISS':
        filter_name = pupil_kwd
    else:
        filter_name = filter_kwd

    # select a subset of the brightest sources using source catalog
    if brightest_n is not None:
        source_catalog = Table.read(input_model.meta.source_catalog, format='ascii.ecsv')
        source_catalog.sort("isophotal_abmag", reverse=False) #magnitudes in ascending order, since brighter is smaller mag number
        selected_IDs = list(source_catalog["label"])[:brightest_n]
    else:
        selected_IDs = None

    obs = Observation(image_names, seg_model, grism_wcs, filter_name,
                      boundaries=[0, 2047, 0, 2047], offsets=[xoffset, yoffset], max_cpu=ncpus,
                      ID=selected_IDs)
    
    good_slits = [slit for slit in output_model.slits if slit.source_id in obs.IDs]
    output_model = datamodels.MultiSlitModel()
    output_model.slits.extend(good_slits)
    log.info(f"Simulating only the brightest {brightest_n} sources")


    simul_all = None
    for order in spec_orders:

        # Load lists of wavelength ranges and flux cal info
        wavelength_range = waverange.get_wfss_wavelength_range(filter_name, [order])
        wmin = wavelength_range[order][0]
        wmax = wavelength_range[order][1]
        log.debug(f"wmin={wmin}, wmax={wmax} for order {order}")
        sens_waves, sens_response = get_photom_data(photom, filter_kwd, pupil_kwd, order)

        # Create simulated grism image for each order and sum them up
        log.info(f"Creating full simulated grism image for order {order}")
        obs.disperse_all(order, wmin, wmax, sens_waves, sens_response)
        if simul_all is None:
            simul_all = obs.simulated_image
        else:
            simul_all += obs.simulated_image

    # Save the full-frame simulated grism image
    simul_model = datamodels.ImageModel(data=simul_all)
    simul_model.update(input_model, only="PRIMARY")

    simul_slit_sids = np.array(obs.simul_slits_sid)
    simul_slit_orders = np.array(obs.simul_slits_order)

    # Loop over all slits/sources to subtract contaminating spectra
    log.info("Creating contamination image for each individual source")
    contam_model = datamodels.MultiSlitModel()
    contam_model.update(input_model)
    slits = []
    for slit in output_model.slits:

        # Retrieve simulated slit for this source only
        sid = slit.source_id
        order = slit.meta.wcsinfo.spectral_order
        good = (simul_slit_sids == sid) * (simul_slit_orders == order)
        if not any(good):
            log.warning(f"Source {sid} order {order} requested by input slit model \
                        but not found in simulated slits")
            continue
        else:
            print('Subtracting contamination for source', sid, 'order', order)
        good_idx = np.where(good)[0][0]
        this_simul = obs.simul_slits.slits[good_idx]

        # cut out this source's contamination from the full simulated image
        fullframe_sim = np.zeros(obs.dims)
        y0 = this_simul.ystart 
        x0 = this_simul.xstart 
        fullframe_sim[y0:y0 + this_simul.ysize, x0:x0 + this_simul.xsize] = this_simul.data
        contam = simul_all - fullframe_sim

        # Create a cutout of the contam image that matches the extent
        # of the source slit
        x1 = slit.xstart - 1
        y1 = slit.ystart - 1
        cutout = contam[y1:y1 + slit.ysize, x1:x1 + slit.xsize]
        new_slit = datamodels.SlitModel(data=cutout)
        # TO DO:
        # not sure if the slit metadata is getting transferred properly
        copy_slit_info(slit, new_slit) 
        slits.append(new_slit)

        # Subtract the cutout from the source slit
        slit.data -= cutout

    # Save the contamination estimates for all slits
    contam_model.slits.extend(slits)
    print('number of slits in contam model', len(contam_model.slits))
    print('number of slits in output model', len(output_model.slits))
    print('number of slits in simul model', len(obs.simul_slits.slits))

    # at what point does the output model get updated with the contamination-corrected data?

    # Set the step status to COMPLETE
    output_model.meta.cal_step.wfss_contam = 'COMPLETE'

    return output_model, simul_model, contam_model, obs.simul_slits


def copy_slit_info(input_slit, output_slit):

    """Copy meta info from one slit to another.

    Parameters
    ----------
    input_slit : SlitModel
        Input slit model from which slit-specific info will be copied

    output_slit : SlitModel
        Output slit model to which slit-specific info will be copied

    """
    output_slit.name = input_slit.name
    output_slit.xstart = input_slit.xstart
    output_slit.ystart = input_slit.ystart
    output_slit.xsize = input_slit.xsize
    output_slit.ysize = input_slit.ysize
    output_slit.source_id = input_slit.source_id
    output_slit.source_type = input_slit.source_type
    output_slit.source_xpos = input_slit.source_xpos
    output_slit.source_ypos = input_slit.source_ypos
    output_slit.meta.wcsinfo.spectral_order = input_slit.meta.wcsinfo.spectral_order
    output_slit.meta.wcsinfo.dispersion_direction = input_slit.meta.wcsinfo.dispersion_direction
    output_slit.meta.wcs = input_slit.meta.wcs
