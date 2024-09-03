import numpy as np
import json
from pathlib import Path

from scipy import optimize
from scipy.ndimage import fourier_shift

from stdatamodels.jwst.datamodels import CubeModel
from jwst.datamodels import ModelLibrary

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def align_fourierLSQ(reference, target, mask=None):
    '''LSQ optimization with Fourier shift alignment

    Parameters
    ----------

        reference : numpy.ndarray
            A 2D (``NxK``) image to be aligned to

        target : numpy.ndarray
            A 2D (``NxK``) image to align to reference

        mask : numpy.ndarray, None
            A 2D (``NxK``) image indicating pixels to ignore when
            performing the minimization. The masks acts as
            a weighting function in performing the fit.

    Returns
    -------

        results : numpy.ndarray
            A 1D vector containing (`xshift`, `yshift`, `beta`) values from
            LSQ optimization, where `xshift` and `yshift` are the misalignment
            of target from reference and `beta` is the fraction by which the
            target intensity must be reduced to match the intensity
            of the reference.
    '''

    init_pars = [0., 0., 1.]
    results, _ = optimize.leastsq(shift_subtract, init_pars,
                                  args=(reference, target, mask))
    return results


def shift_subtract(params, reference, target, mask=None):
    '''Use Fourier Shift theorem for subpixel shifts.

    Parameters
    ----------

        params : tuple
            xshift, yshift, beta

        reference : numpy.ndarray
            See align_fourierLSQ

        target : numpy.ndarray
            See align_fourierLSQ

        mask : numpy.ndarray, None
            See align_fourierLSQ

    Returns
    -------

        1D numpy.ndarray of target-reference residual after
        applying shift and intensity fraction.

    '''
    shift = params[:2]
    beta = params[2]

    offset = fourier_imshift(reference, shift)

    if mask is not None:
        return ((target - beta * offset) * mask).ravel()
    else:
        return (target - beta * offset).ravel()


def fourier_imshift(image, shift):
    '''  Shift an image by use of Fourier shift theorem

    Parameters
    ----------

        image : numpy.ndarray
            A 2D (``NxK``) or 3D (``LxNxK``) image.

        shift : numpy.ndarray
            A 1D or 2D array of shape ``Lx2`` containing pixel values by which
            to shift image slices in the X and Y directions.

    Returns
    -------

        offset : numpy.ndarray
            Shifted image

    '''
    ndim = len(image.shape)

    if ndim == 2:
        shift = np.asanyarray(shift)[:2]
        offset_image = fourier_shift(
            np.fft.fftn(image),
            shift[::-1]
        )
        offset = np.fft.ifftn(offset_image).real

    elif ndim == 3:
        nslices = image.shape[0]
        shift = np.asanyarray(shift)[:, :2]
        if shift.shape[0] != nslices:
            raise ValueError("The number of provided shifts must be equal "
                             "to the number of slices in the input image.")

        offset = np.empty_like(image, dtype=float)
        for k in range(nslices):
            offset[k] = fourier_imshift(image[k], shift[k])

    else:
        raise ValueError("Input image must be either a 2D or a 3D array.")

    return offset


def align_array(reference, target, mask=None, return_aligned=True):
    """
    Computes shifts between target image (or image "slices") and the reference
    image and re-aligns input images to the target.

    Parameters
    ----------

    reference : numpy.ndarray
        A 2D (``NxK``) reference image to which input images will be aligned.

    target : numpy.ndarray
        A 2D (``NxK``) or 3D (``MxNxK`` first index used to select slices)
        image(s) that need to be aligned to the reference image.

    mask : numpy.ndarray, None
        A 2D (``NxK``) image indicating pixels to ignore when performing the
        minimization. The masks acts as a weighting function in performing
        the fit.

    Returns
    -------

        A tuple containing the aligned image (2D or 3D image of the same shape
        as input target image) and a 1D vector of three elements in the case
        of 2D input `target` image of (xshift, yshift, beta) values from
        LSQ optimization (see :py:func:`align_fourierLSQ` for details) for each
        slice in the `target` array.

    """

    if len(target.shape) == 2:
        shifts = align_fourierLSQ(reference, target, mask=mask)
        if return_aligned:
            aligned = fourier_imshift(target, -shifts)

    elif len(target.shape) == 3:
        nslices = target.shape[0]
        shifts = np.empty((nslices, 3), dtype=float)
        if return_aligned:
            aligned = np.empty_like(target)

        for m in range(nslices):
            sh = align_fourierLSQ(reference, target[m], mask=mask)
            shifts[m, :] = sh
            if return_aligned:
                aligned[m, :, :] = fourier_imshift(target[m], -sh)

    else:
        raise ValueError("Input target image must be either a 2D or 3D array.")

    if not return_aligned:
        return shifts
    return aligned, shifts


def align_models(reference, target, mask, in_memory=False, output_dir="psfalign"):
    """
    Computes shifts between target image (or image "slices") and the reference
    image and re-aligns target images to the reference.

    Parameters
    ----------

    reference : CubeModel
        3D (``LxNxK`` first index used to select 2D slices)
        reference image to which target images will be aligned.

    target : CubeModel
        3D (``MxNxK`` first index used to select 2D slices)
        image(s) that need to be aligned to the reference image.

    mask : ImageModel, None
        A 2D (``NxK``) image indicating pixels to ignore when performing the
        minimization. Mask acts as a weighting function in performing
        the fit.

    in_memory : bool
        If True, the output psfalign library is kept in memory. If False, the output
        library is saved to disk in on_disk mode

    output_dir : str
        The directory where the output psfalign library will be saved if in_memory
        is False.

    Returns
    -------

        A ModelLibrary containing aligned copies of the input ``target``
        cubes aligned to each slice in the input ``reference`` cube.

    """

    # Get the number of integrations in the science exposure
    nrefslices = reference.data.shape[0]

    # Create output QuadModel of required dimensions
    cube_shape = (target.shape[0], target.shape[1], target.shape[2])

    # Compute the shifts of the PSF ("target") images relative to
    # the science ("reference") image in the first integration
    shifts = align_array(
        reference.data[0].astype(np.float64),
        target.data.astype(np.float64),
        mask=mask.data, return_aligned=False)

    # Loop over all integrations of the science exposure
    tmpdir_psfalign = Path(output_dir)
    tmpdir_psfalign.mkdir(exist_ok=True)
    log.info(f"Creating psfalign library, {nrefslices} models each of shape %s", cube_shape)
    psfalign_files = []
    for k in range(nrefslices):

        # Apply the shifts to the PSF images
        data = fourier_imshift(
            target.data.astype(np.float64),
            -shifts)
        
        output_cube = CubeModel(data=data)
        output_cube.update(target)

        # Apply the same shifts to the PSF error arrays, if they exist
        if target.err is not None:
            output_cube.err = fourier_imshift(
                target.err.astype(np.float64),
                -shifts)
            
        output_cube.meta.cal_step.align_psfs = 'COMPLETE'
        if not in_memory:
            fname = "psfalign_" + str(k) + ".fits"
            output_cube.save(tmpdir_psfalign / fname)
            psfalign_files.append(fname)
        else:
            psfalign_files.append(output_cube)

        # TODO: in the future we need to add shifts and other info (such as
        # slice ID from the reference image to which target was aligned)
        # to output cube metadata (or property).
    
    if not in_memory:
    # turn the list of file names into a ModelLibrary
        psfalign_members = [{'expname': fname, 'exptype': 'science'} for fname in psfalign_files]
        psfalign_asn = {"products":[{"name":"coron3_psfalign","members":psfalign_members}]}
        asn_path = tmpdir_psfalign / "asn.json"
        asn_path.write_text(json.dumps(psfalign_asn))
        output_model = ModelLibrary(asn_path, on_disk=not in_memory)
    else:
        output_model = ModelLibrary(psfalign_files)

    return output_model
