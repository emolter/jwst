"""

Unit tests for ami_analyze

"""
import numpy as np
import math

import pytest

from stdatamodels.jwst import datamodels

from jwst.ami import utils, leastsqnrm
from jwst.ami.leastsqnrm import closure_amplitudes
from jwst.ami.leastsqnrm import redundant_cps
from jwst.ami.leastsqnrm import populate_symmamparray
from jwst.ami.leastsqnrm import populate_antisymmphasearray
from jwst.ami.leastsqnrm import tan2visibilities
from jwst.ami.analyticnrm2 import interf, psf, phasor, asf_hex

from numpy.testing import assert_allclose


# ---------------------------------------------------------------
# utils module tests:
#
def test_utils_rebin():
    """Test of rebin() and krebin() in utils module"""
    arr = np.arange(24).reshape((3, 8)) / 10.0
    rc = tuple((2, 2))

    binned_arr = utils.rebin(arr, rc)

    true_arr = np.array([[5.1, 6.3, 7.5, 8.7]])
    assert_allclose(binned_arr, true_arr)


def test_utils_makeA():
    """Test of makeA in utils module"""
    nh = 4  # number of holes
    arr = utils.make_a(nh)

    true_arr = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    assert_allclose(arr, true_arr)


def test_utils_fringes2pistons():
    """Test of fringes2pistons in utils module"""
    fringephases = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    nholes = 5

    result = utils.fringes2pistons(fringephases, nholes)

    true_result = np.array([-0.02, -0.034, -0.02, 0.014, 0.06])
    assert_allclose(result, true_result)


def test_utils_rcrosscorrelate():
    """Test of rcrosscorrelate() in utils module"""
    a = np.array(
        [
            [2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0, 17.0],
        ]
    )

    b = np.array(
        [
            [-5.0, -4.0, -3.0, -2.0],
            [-1.0, 0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
    )

    result = utils.rcrosscorrelate(a, b)

    true_result = np.array(
        [
            [0.19865015, 0.20767971, 0.23476836, 0.20767971],
            [0.34312299, 0.35215254, 0.3792412, 0.35215254],
            [0.77654151, 0.78557106, 0.81265972, 0.78557106],
            [0.34312299, 0.35215254, 0.3792412, 0.35215254],
        ]
    )
    assert_allclose(result, true_result)


def test_utils_crosscorrelate():
    """Test of crosscorrelate() in utils module"""
    a = np.array(
        [
            [2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0, 17.0],
        ]
    )

    b = np.array(
        [
            [-5.0, -4.0, -3.0, -2.0],
            [-1.0, 0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
    )

    result = utils.crosscorrelate(a, b)

    true_result = np.array(
        [
            [176.0 + 0.0j, 184.0 + 0.0j, 208.0 + 0.0j, 184.0 + 0.0j],
            [304.0 + 0.0j, 312.0 + 0.0j, 336.0 + 0.0j, 312.0 + 0.0j],
            [688.0 + 0.0j, 696.0 + 0.0j, 720.0 + 0.0j, 696.0 + 0.0j],
            [304.0 + 0.0j, 312.0 + 0.0j, 336.0 + 0.0j, 312.0 + 0.0j],
        ]
    )
    assert_allclose(result, true_result)


# ---------------------------------------------------------------
# leastsqnrm module tests:
#


def test_leastsqnrm_replacenan():
    """Test of replacenan() in leastsqnrm module.
    Replace singularities encountered in the analytical hexagon Fourier
    transform with the analytically derived limits. (pi/4)
    """
    arr = np.array([1.0, 5.6, np.nan, 5.3])

    rep_arr = leastsqnrm.replacenan(arr)

    true_rep_arr = np.array([1.0, 5.6, math.pi / 4.0, 5.3])
    assert_allclose(rep_arr, true_rep_arr)


def test_leastsqnrm_closure_amplitudes():
    """Test of closure_amplitudes() in leastsqnrm module.
    Calculate the closure amplitudes.
    """
    amps = np.array([0.1, 0.2, 0.3, 1.0, 0.9, 0.5, 1.1, 0.7, 0.1, 1.0])

    n = 5  # number of holes

    cas = closure_amplitudes(amps, n=n)

    true_cas = np.array([0.7, 0.04545455, 0.3030303, 6.66666667, 18.0])
    assert_allclose(cas, true_cas, atol=1e-7)


def test_leastsqnrm_redundant_cps():
    """Test of redundant_cps in leastsqnrm module.
    Calculate closure phases for each set of 3 holes.
    """
    n = 7  # number of holes
    deltaps = np.array(
        [
            0.1,
            -0.2,
            0.3,
            0.2,
            0.05,
            -0.7,
            -0.05,
            0.7,
            0.1,
            0.02,
            -0.5,
            -0.05,
            0.7,
            0.1,
            0.3,
            0.4,
            -0.2,
            -0.3,
            0.2,
            0.5,
            0.3,
        ]
    )

    cps = redundant_cps(deltaps, n=n)

    true_cps = np.array(
        [
            0.25,
            0.5,
            0.0,
            0.07,
            0.3,
            -0.55,
            0.3,
            -0.15,
            0.8,
            0.5,
            0.05,
            0.7,
            0.35,
            1.4,
            1.05,
            -0.8,
            0.55,
            0.03,
            0.75,
            1.0,
            0.48,
            0.9,
            0.28,
            1.1,
            0.82,
            -0.35,
            -0.35,
            -0.65,
            0.8,
            0.9,
            0.1,
            0.8,
            1.2,
            0.4,
            0.0,
        ]
    )
    assert_allclose(cps, true_cps, atol=1e-8)


def test_leastsqnrm_populate_symmamparray():
    """Test of populate_symmamparray in leastsqnrm module.
    Populate the symmetric fringe amplitude array.
    """
    amps = np.array([0.1, 0.2, 0.3, 0.2, 0.05, 0.7, 0.3, 0.1, 0.2, 0.8])
    n = 5

    arr = populate_symmamparray(amps, n=n)

    true_arr = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.2],
            [0.1, 0.0, 0.05, 0.7, 0.3],
            [0.2, 0.05, 0.0, 0.1, 0.2],
            [0.3, 0.7, 0.1, 0.0, 0.8],
            [0.2, 0.3, 0.2, 0.8, 0.0],
        ]
    )
    assert_allclose(arr, true_arr, atol=1e-8)


def test_leastsqnrm_populate_antisymmphasearray():
    """Test of populate_antisymmphasearray in leastsqnrm module.
    Populate the antisymmetric fringe phase array.
    """
    deltaps = np.array([0.1, 0.2, 0.3, 0.2, 0.05, 0.7, 0.3, 0.1, 0.2, 0.8])
    n = 5

    arr = populate_antisymmphasearray(deltaps, n=n)

    true_arr = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.2],
            [-0.1, 0.0, 0.05, 0.7, 0.3],
            [-0.2, -0.05, 0.0, 0.1, 0.2],
            [-0.3, -0.7, -0.1, 0.0, 0.8],
            [-0.2, -0.3, -0.2, -0.8, 0.0],
        ]
    )
    assert_allclose(arr, true_arr, atol=1e-8)


def test_leastsqnrm_tan2visibilities():
    """Test of tan2visibilities in leastsqnrm module.
    From the solution to the fit, calculate the fringe amplitude and phase.
    """
    test_res = []  # to accumulate subtest comparisons

    coeffs = np.array([1.0, 0.2, -0.3, -0.1, 0.4, 0.2, -0.5, -0.1, 0.2, 0.4])

    amp, delta = tan2visibilities(coeffs)

    true_amp = np.array([0.36055513, 0.41231056, 0.53851648, 0.2236068])
    test_res.append(np.allclose(amp, true_amp, rtol=1.0e-7))

    true_delta = np.array([-0.98279372, 1.81577499, -1.19028995, 2.03444394])
    test_res.append(np.allclose(delta, true_delta, rtol=1.0e-7))

    assert np.all(test_res)


def test_leastsqnrm_multiplyenv():
    """Test of multiplyenv in leastsqnrm module.
    Multiply the envelope by each fringe 'image'.
    """
    env = np.array(
        [
            [1.00e-06, 1.00e-06, 1.01e-04],
            [1.00e-06, 1.00e-06, 1.01e-04],
            [1.10e-05, 1.10e-05, 1.00e-06],
        ]
    )

    ft = np.array(
        [
            [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
            [[0.3, 0.3, 0.3], [0.3, 0.3, 0.4], [0.3, 0.3, 0.3]],
            [[-0.4, -0.4, -0.4], [-0.4, -0.4, -0.6], [-0.4, -0.4, -0.9]],
            [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [1.0, 0.8, 1.3]],
        ]
    )

    # function is expecting a list, so make it one
    fringeterms = list(ft)

    full = leastsqnrm.multiplyenv(env, fringeterms)

    true_full = np.array(
        [
            [
                [4.00e-06, 3.00e-07, -4.00e-07, 8.00e-07, 1.00e00],
                [4.00e-06, 3.00e-07, -4.00e-07, 8.00e-07, 1.00e00],
                [4.04e-04, 3.03e-05, -4.04e-05, 8.08e-05, 1.00e00],
            ],
            [
                [4.00e-06, 3.00e-07, -4.00e-07, 8.00e-07, 1.00e00],
                [4.00e-06, 3.00e-07, -4.00e-07, 8.00e-07, 1.00e00],
                [4.04e-04, 4.04e-05, -6.06e-05, 8.08e-05, 1.00e00],
            ],
            [
                [4.40e-05, 3.30e-06, -4.40e-06, 1.10e-05, 1.00e00],
                [4.40e-05, 3.30e-06, -4.40e-06, 8.80e-06, 1.00e00],
                [4.00e-06, 3.00e-07, -9.00e-07, 1.30e-06, 1.00e00],
            ],
        ]
    )

    assert_allclose(full, true_full, atol=1e-8)


# ---------------------------------------------------------------
# analyticnrm2 module tests:
#
def test_analyticnrm2_psf(setup_sf):
    """Test of psf() in the analyticnrm2 module"""
    pixel, fov, oversample, ctrs, d, lam, phi, psf_offset, aff_obj = setup_sf
    shape = "hex"

    computed_psf = psf(pixel, fov, oversample, ctrs, d, lam, phi, psf_offset, aff_obj, shape=shape)

    true_psf = np.array(
        [
            [1.14249135, 0.65831385, 0.45119464, 0.66864436, 1.10501352, 2.04851966],
            [2.2221824, 0.62716999, 0.87062628, 1.97855142, 1.72666739, 0.28363866],
            [4.37562298, 2.64951632, 6.40126821, 12.22910105, 13.17326852, 7.49323549],
            [5.93942383, 4.58894785, 12.68235611, 24.87843624, 29.17900067, 20.64525322],
            [5.38441424, 3.73680387, 13.26524812, 28.96518165, 36.75, 28.96518165],
            [3.98599305, 1.08124031, 7.38628086, 20.64525322, 29.17900067, 24.87843625],
        ]
    )

    assert_allclose(computed_psf, true_psf, atol=1e-7)


def test_analyticnrm2_asf_hex(setup_sf):
    """Test of asf_hex() in the analyticnrm2 module FOR HEX"""
    pixel, fov, oversample, _ctrs, d, lam, _phi, psf_offset, aff_obj = setup_sf

    asf = asf_hex(pixel, fov, oversample, d, lam, psf_offset, aff_obj)

    true_asf = np.array(
        [
            [
                0.82125698 + 7.84095011e-16j,
                0.83091456 + 2.48343013e-14j,
                0.83785899 - 2.49800181e-16j,
                0.84204421 - 1.80411242e-16j,
                0.8434424 - 2.91433544e-16j,
                0.84204424 - 1.24900090e-16j,
            ],
            [
                0.83091447 + 4.09394740e-16j,
                0.84064761 + 1.38777878e-15j,
                0.8476463 + 1.29063427e-15j,
                0.85186417 - 6.17561557e-16j,
                0.85327325 + 2.98372438e-16j,
                0.85186418 + 1.90125693e-15j,
            ],
            [
                0.83785894 + 1.68268177e-16j,
                0.84764629 + 1.07552856e-16j,
                0.8546839 + 6.38378239e-16j,
                0.8589252 - 1.65145675e-15j,
                0.8603421 - 9.29811783e-16j,
                0.8589252 + 1.15185639e-15j,
            ],
            [
                0.84204421 - 6.59194921e-17j,
                0.85186417 - 6.70470623e-16j,
                0.8589252 + 8.91214186e-16j,
                0.86318061 - 3.46944695e-16j,
                0.86460222 + 2.08166817e-17j,
                0.86318061 - 5.34294831e-16j,
            ],
            [
                0.84344243 + 2.28983499e-16j,
                0.85327326 + 2.98719383e-15j,
                0.8603421 + 5.02722863e-15j,
                0.86460222 + 5.48866508e-15j,
                0.8660254 + 0.00000000e00j,
                0.86460222 + 5.29611077e-15j,
            ],
            [
                0.84204425 - 1.48492330e-15j,
                0.85186418 + 6.03683770e-16j,
                0.8589252 + 5.68989300e-16j,
                0.86318061 + 2.77555756e-16j,
                0.86460222 - 1.72431514e-15j,
                0.86318061 - 5.54070678e-15j,
            ],
        ]
    )

    assert_allclose(asf, true_asf, atol=1e-7)


def test_analyticnrm2_interf(setup_sf):
    """Test of interf() in the analyticnrm2 module"""
    ASIZE = 4
    kx = np.arange(ASIZE * ASIZE).reshape((ASIZE, ASIZE))
    ky = np.arange(ASIZE * ASIZE).reshape((ASIZE, ASIZE))
    vv = np.arange(ASIZE)

    for ii in np.arange(ASIZE):
        kx[:, ii] = vv
        ky[ii, :] = vv

    # Clean up any attributes that may have been added earlier
    for kk in list((interf.__dict__).keys()):
        delattr(interf, kk)

    pixel, fov, oversample, ctrs, d, lam, phi, centering, aff_obj = setup_sf

    pitch = pixel / float(oversample)

    interf.lam = lam
    interf.offx = 0.5
    interf.offy = 0.5
    interf.ctrs = ctrs
    interf.d = d
    interf.phi = phi
    interf.pitch = pixel / float(oversample)

    c = (ASIZE / 2.0, ASIZE / 2)
    interf.c = (ASIZE / 2.0, ASIZE / 2)

    interference = interf(kx, ky, ctrs=ctrs, phi=phi, lam=lam, pitch=pitch, c=c, affine2d=aff_obj)

    true_interference = np.array(
        [
            [
                2.6870043 + 1.24219632j,
                4.01721904 + 0.66189711j,
                4.2132531 + 0.21372447j,
                3.18675131 - 0.03818252j,
            ],
            [
                3.8517604 + 1.53442862j,
                5.71582424 + 0.84829672j,
                6.24380079 + 0.2201634j,
                5.25470657 - 0.31113349j,
            ],
            [
                4.02194801 + 1.32112798j,
                6.1888738 + 0.66733046j,
                7.0 + 0.0j,
                6.1888738 - 0.66733046j,
            ],
            [
                3.07194559 + 0.75829976j,
                5.25470657 + 0.31113349j,
                6.24380079 - 0.2201634j,
                5.71582424 - 0.84829672j,
            ],
        ]
    )

    assert_allclose(interference, true_interference, atol=1e-7)


def test_analyticnrm2_phasor():
    """Test of phasor() in the analyticnrm2 module"""
    ASIZE = 4
    kx = np.arange(ASIZE * ASIZE).reshape((ASIZE, ASIZE))

    for ii in np.arange(ASIZE):
        kx[:, ii] = ii

    ky = kx.transpose()

    hx = 0.06864653345335156
    hy = -2.6391073592116028

    lam = 2.3965000082171173e-06
    phi = 0.0
    pitch = 1.0375012775744072e-07

    aff_obj = utils.Affine2d(rotradccw=0.4)

    result = phasor(kx, ky, hx, hy, lam, phi, pitch, aff_obj)

    true_result = np.array(
        [
            [
                1.0 + 0.0j,
                0.96578202 + 0.25935515j,
                0.86546981 + 0.50096108j,
                0.70592834 + 0.70828326j,
            ],
            [
                0.78476644 + 0.61979161j,
                0.59716716 + 0.80211681j,
                0.36870018 + 0.92954837j,
                0.11500085 + 0.99336539j,
            ],
            [
                0.23171672 + 0.97278331j,
                -0.02850852 + 0.99959355j,
                -0.28678275 + 0.95799564j,
                -0.52543073 + 0.85083638j,
            ],
            [
                -0.42107943 + 0.90702377j,
                -0.64191223 + 0.76677812j,
                -0.81881514 + 0.57405728j,
                -0.93968165 + 0.34205027j,
            ],
        ]
    )

    assert_allclose(result, true_result, atol=1e-7)

# ---------------------------------------------------------------
# utility functions:


@pytest.fixture
def setup_sf():
    """Initialize values for these parameters needed for the analyticnrm2 tests.

    Returns
    -------
    pixel (optional, via **kwargs) : float
        pixel scale

    fov : integer
        number of detector pixels on a side

    oversample : integer
        oversampling factor

    ctrs : float, float
        coordinates of hole centers

    d : float
        hole diameter

    lam : float
        wavelength

    phi : float
        distance of fringe from hole center in units of waves

    centering : string
        if set to 'PIXELCENTERED' or unspecified, the offsets will be set to
        (0.5,0.5); if set to 'PIXELCORNER', the offsets will be set to
        (0.0,0.0).

    aff : Affine2d object
        Affine2d object

    """
    pixel = 3.1125038327232215e-07
    fov = 2
    oversample = 3
    ctrs = np.array(
        [
            [0.06864653, -2.63910736],
            [-2.28553695, -0.05944972],
            [2.31986022, -1.26010406],
            [-2.31986022, 1.26010406],
            [-1.19424838, 1.94960579],
            [2.25121368, 1.3790035],
            [1.09127858, 2.00905525],
        ]
    )
    d = 0.8
    lam = 2.3965000082171173e-06
    phi = np.zeros(7, dtype=np.float32)
    centering = (0.5, 0.5)
    aff_obj = utils.Affine2d(rotradccw=0.4)

    return pixel, fov, oversample, ctrs, d, lam, phi, centering, aff_obj


def setup_hexee():
    """Initialize values for parameters needed for the hexee tests.

    Returns
    -------
    xi : 2D float array
        hexagon's coordinate center at center of symmetry, along flat edge

    eta : 2D float array
        hexagon's coordinate center at center of symmetry, normal to xi;
        (not currently used)

    c (optional, via **kwargs) : tuple(float, float)
        coordinates of center

    pixel (optional, via **kwargs) : float
        pixel scale

    d (optional, via **kwargs) : float
        flat-to-flat distance across hexagon

    lam : (optional, via **kwargs) : float
        wavelength

    minus : (optional, via **kwargs) boolean
        if set, use flipped sign of xi in calculation
    """
    xdim, ydim = 3, 3
    xi = np.zeros(ydim * xdim).reshape((ydim, xdim))
    eta = np.zeros(ydim * xdim).reshape((ydim, xdim))

    for ii in range(ydim):
        xi[ii, :] = ii
        eta[:, ii] = ii

    kwargs = {
        'd': 0.8,
        'c': (28.0, 28.0),
        'lam': 2.3965000082171173e-06,
        'pixel': 1.0375012775744072e-07,
        'minus': False,
    }

    return xi, eta, kwargs


def create_throughput(nelem):
    """Create a symmetric dummy throughput function that has values near
    0 on the wings and near 1 at the center.
    """
    ctr = int(nelem / 2.0)

    lower_half = [2.0 / (1.0 + math.e ** (-5.0 * i / ctr)) - 1.0 for i in range(ctr)]

    throughput = np.zeros(nelem, dtype=np.float32)
    throughput[:ctr] = lower_half
    throughput[ctr:] = lower_half[::-1]  # mirror image for upper half

    return throughput
