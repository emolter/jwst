"""
    Stack individual coronagraphic PSF reference images into a cube model.

:Authors: Howard Bushouse

"""

import numpy as np
from stdatamodels.jwst import datamodels

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def make_cube(input_models):
    """
    make_cube: Stack all of the integrations from multiple PSF
    reference exposures into a single CubeModel, for use in the
    coronagraphic alignment and PSF-subtraction steps.
    """

    # Loop over all the inputs to find the total number of integrations
    nints = 0
    with input_models:
        for i, model in enumerate(input_models):
            if i == 0:
                nrows_ref, ncols_ref = model.shape[-2:]
            nints += model.shape[0]
            nrows, ncols = model.shape[-2:]
            if nrows != nrows_ref or ncols != ncols_ref:
                raise ValueError('All PSF exposures must have the same x/y dimensions!')
            input_models.shelve(model, i, modify=False)

    # Create empty output data arrays of the appropriate dimensions
    outdata = np.zeros((nints, nrows, ncols), dtype=np.float64) #q: do these need to be 64 bit floats?
    outerr = np.zeros((nints, nrows, ncols), dtype=np.float64)
    outdq = np.zeros((nints, nrows, ncols), dtype=np.uint32)

    # Loop over the input images, copying the data arrays
    # into the output arrays
    nints = 0
    output_model = datamodels.CubeModel()
    with input_models:
        for i, model in enumerate(input_models):
            log.info(' Adding psf member %d to output stack', i + 1)
            sz = model.shape[0]
            outdata[nints:nints+sz] = model.data
            outerr[nints:nints+sz] = model.err
            outdq[nints:nints+sz] = model.dq
            nints += sz
            input_models.shelve(model, i, modify=False)

    # Create the output Cube model
    output_model.data = outdata
    output_model.err = outerr
    output_model.dq = outdq
    del outdata, outerr, outdq
    output_model.update(model)  # copy meta data from last input
    del model

    return output_model
