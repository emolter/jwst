import numpy as np

from jwst.wfss_geom import disperse
from jwst.wfss_geom.wfss_geom import _combine_geom_chunks


def test_disperse_geometry_pixel_index_pointer(monkeypatch):
    n_wavelengths = 2
    n_pixels = 3
    grism_x = np.array([[1.5, 3.5, 20.5], [2.5, 4.5, 20.5]])
    grism_y = np.array([[1.5, 1.5, 20.5], [2.5, 2.5, 20.5]])

    class Transform:
        outputs = ("x", "y")

        def __or__(self, other):
            return self

    class GrismWCS:
        def get_transform(self, *args):
            return Transform()

    def direct_image_wcs(x, y, with_bounding_box):
        return x, y

    def disperse_onto_grism(x, y, sky_to_img, img_to_grism, lambdas, order, trace_pdt=None):
        return grism_x, grism_y, lambdas

    monkeypatch.setattr(disperse, "_disperse_onto_grism", disperse_onto_grism)

    source_ids = np.array([10, 20, 30])
    input_x = np.array([4, 8, 12])
    input_y = np.array([5, 9, 13])
    geometry = disperse.disperse(
        input_x,
        input_y,
        source_ids,
        order=1,
        lambdas=np.array([1.0, 2.0]),
        direct_image_wcs=direct_image_wcs,
        grism_wcs=GrismWCS(),
        naxis=(10, 10),
    )

    pointer = geometry["pixel_idx_ptr"]
    wavelength_index = geometry["wavelength_index"]
    assert pointer.shape == (n_pixels + 1,)
    assert pointer[0] == 0
    assert pointer[-1] == len(wavelength_index)
    assert np.all(np.diff(pointer) >= 0)
    assert pointer[-1] == len(geometry["grism_xy"]) == len(geometry["fractional_area"])
    assert geometry["source_id"].shape == (n_pixels,)

    # The third input pixel's trace is fully off-detector and has an empty slice.
    assert pointer[-2] == pointer[-1]
    np.testing.assert_array_equal(geometry["input_xy"], np.column_stack((input_x, input_y)))

    for pixel_idx, (start, stop) in enumerate(zip(pointer[:-1], pointer[1:], strict=True)):
        assert geometry["source_id"][pixel_idx] == source_ids[pixel_idx]
        assert np.all(
            (wavelength_index[start:stop] >= 0) & (wavelength_index[start:stop] < n_wavelengths)
        )
        if pixel_idx < n_pixels - 1:
            np.testing.assert_array_equal(
                np.unique(wavelength_index[start:stop]), np.arange(n_wavelengths)
            )


def test_combine_geometry_chunks_rebases_pixel_index_pointer():
    first = {
        "input_xy": np.array([[1, 2], [3, 4]]),
        "source_id": np.array([10, 10]),
        "pixel_idx_ptr": np.array([0, 2, 2]),
        "grism_xy": np.array([[5, 6], [7, 8]]),
        "wavelength_index": np.array([0, 1]),
        "fractional_area": np.array([0.25, 0.75]),
    }
    second = {
        "input_xy": np.array([[9, 10], [11, 12]]),
        "source_id": np.array([20, 20]),
        "pixel_idx_ptr": np.array([0, 1, 3]),
        "grism_xy": np.array([[13, 14], [15, 16], [17, 18]]),
        "wavelength_index": np.array([1, 0, 1]),
        "fractional_area": np.array([0.5, 0.2, 0.8]),
    }

    pdt = _combine_geom_chunks([first, second], wavelength=np.array([1.0, 2.0]))

    np.testing.assert_array_equal(pdt["pixel_idx_ptr"], [0, 2, 2, 3, 5])
    np.testing.assert_array_equal(pdt["input_xy"], [[1, 2], [3, 4], [9, 10], [11, 12]])
    np.testing.assert_array_equal(pdt["source_id"], [10, 10, 20, 20])
    np.testing.assert_array_equal(pdt["grism_xy"], [[5, 6], [7, 8], [13, 14], [15, 16], [17, 18]])
    np.testing.assert_array_equal(pdt["wavelength_index"], [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(pdt["fractional_area"], [0.25, 0.75, 0.5, 0.2, 0.8])


def test_combine_geometry_chunks_empty():
    pdt = _combine_geom_chunks([], wavelength=np.array([1.0, 2.0]))

    assert pdt["input_xy"].shape == (0, 2)
    assert pdt["source_id"].shape == (0,)
    np.testing.assert_array_equal(pdt["pixel_idx_ptr"], [0])
    assert pdt["grism_xy"].shape == (0, 2)
    assert pdt["wavelength_index"].shape == (0,)
    assert pdt["fractional_area"].shape == (0,)

    empty_chunk = {
        "input_xy": np.array([[1, 2], [3, 4]]),
        "source_id": np.array([10, 20]),
        "pixel_idx_ptr": np.zeros(3, dtype=np.intp),
        "grism_xy": np.empty((0, 2), dtype=int),
        "wavelength_index": np.empty(0, dtype=int),
        "fractional_area": np.empty(0, dtype=float),
    }
    pdt = _combine_geom_chunks([empty_chunk], wavelength=np.array([1.0, 2.0]))

    np.testing.assert_array_equal(pdt["pixel_idx_ptr"], [0, 0, 0])
    np.testing.assert_array_equal(pdt["input_xy"], empty_chunk["input_xy"])
    np.testing.assert_array_equal(pdt["source_id"], empty_chunk["source_id"])
