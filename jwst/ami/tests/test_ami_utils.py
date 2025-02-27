
import numpy as np
import pytest

import jwst.ami.utils as utils


@pytest.mark.parametrize("shape, center", [
    ((10, 10), (4.5, 4.5)),
    ((11, 11), (5, 5)),
])
def test_centerpoint(shape, center):
    assert utils.centerpoint(shape) == center


def test_find_centroid():
    arr = np.zeros((30, 30), dtype='f4')
    arr[15, 15] = 1
    assert np.allclose(utils.find_centroid(arr), (0.5, 0.5))
