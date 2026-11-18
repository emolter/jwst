from contextlib import nullcontext
from pathlib import Path

import asdf
import numpy as np
import pytest
from astropy.modeling.models import Identity, Mapping

from jwst.lib.winclip import get_clipped_pixels
from jwst.wfss_geom import disperse, wfss_geom
from jwst.wfss_geom.wfss_geom import Observation
from jwst.wfss_geom.wfss_geom_step import _default_pdt_directory


class GeometryTestWCS:
    def get_transform(self, from_frame, to_frame):
        if from_frame == "world":
            return Identity(4)
        return Mapping((0, 1), n_inputs=4)


def make_observation(tmp_path, max_cpu=1):
    segmentation = np.array([[1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 0]])
    return Observation(
        segmentation,
        GeometryTestWCS(),
        Identity(2),
        boundaries=[0, 29, 0, 9],
        max_cpu=max_cpu,
        max_pixels_per_chunk=5,
        pdt_directory=tmp_path / "test_pdt",
    )


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_storage_uint_boundaries(dtype):
    limit = np.iinfo(dtype).max
    values = np.array([0, limit], dtype=np.int64)
    stored = wfss_geom._storage_uint(values, dtype, "example")
    assert stored.dtype == dtype
    np.testing.assert_array_equal(stored, values)
    for bad_values in [np.array([-1]), np.array([int(limit) + 1], dtype=np.int64)]:
        with pytest.raises(OverflowError, match="example"):
            wfss_geom._storage_uint(bad_values, dtype, "example")


def test_pixel_major_clipping_matches_sorted_wavelength_major(monkeypatch):
    rng = np.random.default_rng(42)
    nlam, n_pixels = 7, 6
    grism_x = rng.uniform(-2, 12, (nlam, n_pixels))
    grism_y = rng.uniform(-2, 12, (nlam, n_pixels))
    grism_x[:, -1] = 50
    grism_x[0, 0] = np.nan
    wavelengths = np.linspace(1.0, 2.0, nlam)
    monkeypatch.setattr(
        disperse,
        "_disperse_onto_grism",
        lambda *args, **kwargs: (grism_x, grism_y, wavelengths),
    )
    actual = disperse.disperse(
        np.arange(n_pixels),
        np.zeros(n_pixels, dtype=int),
        np.arange(1, n_pixels + 1),
        1,
        wavelengths,
        Identity(2),
        GeometryTestWCS(),
        (10, 10),
    )
    xs, ys, areas, index = get_clipped_pixels(grism_x, grism_y, 1, 10, 10, 1.0, 1.0)
    wavelength_index, pixel_index = np.divmod(index, n_pixels)
    ordering = np.argsort(pixel_index, kind="stable")
    np.testing.assert_array_equal(actual["grism_xy"], np.column_stack((xs[ordering], ys[ordering])))
    np.testing.assert_array_equal(actual["fractional_area"], areas[ordering])
    np.testing.assert_array_equal(actual["wavelength_index"], wavelength_index[ordering])
    expected_pointer = np.concatenate(
        ([0], np.cumsum(np.bincount(pixel_index, minlength=n_pixels)))
    )
    np.testing.assert_array_equal(actual["pixel_idx_ptr"], expected_pointer)


def test_source_aware_chunks(tmp_path):
    observation = make_observation(tmp_path)
    chunks = observation.chunk_sources(1, np.array([1.0, 2.0]), max_pixels=5)

    assert [len(args[0]) for args in chunks] == [5, 5, 3]
    assert [np.unique(args[2]).tolist() for args in chunks] == [[1, 2, 3], [3], [3, 4]]
    np.testing.assert_array_equal(np.concatenate([args[0] for args in chunks]), np.arange(13))
    with pytest.raises(ValueError, match="at least 1"):
        observation.chunk_sources(1, np.array([1.0, 2.0]), max_pixels=0)
    assert observation.chunk_sources(1, np.array([1.0, 2.0]), selected_ids=[], max_pixels=5) == []


@pytest.mark.parametrize(
    "source_sizes, expected_sizes",
    [
        ([30, 6000, 40, 100], [5000, 1170]),
        ([30, 5000], [30, 5000]),
        ([3000, 3000], [3000, 3000]),
        ([5000, 1], [5000, 1]),
        ([30, 10000, 40], [5000, 5000, 70]),
    ],
)
def test_only_oversized_sources_can_be_split(tmp_path, source_sizes, expected_sizes):
    labels = np.repeat(np.arange(1, len(source_sizes) + 1), source_sizes)
    segmentation = np.append(labels, 0)[np.newaxis, :]
    observation = Observation(segmentation, GeometryTestWCS(), Identity(2))
    chunks = observation.chunk_sources(1, np.array([1.0, 2.0]), max_pixels=5000)

    assert [len(args[0]) for args in chunks] == expected_sizes
    assert all(len(args[0]) <= 5000 for args in chunks)
    np.testing.assert_array_equal(
        np.concatenate([args[0] for args in chunks]), np.arange(len(labels))
    )
    np.testing.assert_array_equal(np.concatenate([args[2] for args in chunks]), labels)
    for source_id, source_size in enumerate(source_sizes, start=1):
        appearances = sum(np.any(args[2] == source_id) for args in chunks)
        if source_size <= 5000:
            assert appearances == 1
        else:
            assert appearances > 1


@pytest.mark.parametrize("max_cpu", [1, 2])
def test_file_backed_source_manifest(tmp_path, monkeypatch, max_cpu):
    observation = make_observation(tmp_path, max_cpu=max_cpu)
    wavelength = np.array([1.0, 2.0])
    monkeypatch.setattr(wfss_geom, "native_wavelength_grid", lambda *args, **kwargs: wavelength)

    for order in [1, -1]:
        manifest = observation.disperse_order(order, 1.0, 2.0)
        np.testing.assert_array_equal(manifest["wavelength"], wavelength)
        assert set(manifest["sources"]) == {1, 2, 3, 4}
        assert [len(parts) for parts in manifest["sources"].values()] == [1, 1, 3, 1]
        for source_id, filenames in manifest["sources"].items():
            input_pixels = []
            for part, filename in enumerate(filenames):
                assert f"source_{source_id:06d}_order_{order}_part_{part:04d}.asdf" in filename
                with asdf.open(filename) as file:
                    table = file.tree
                    for key, dtype in {
                        "wavelength": np.float32,
                        "fractional_area": np.float32,
                        "grism_xy": np.uint16,
                        "wavelength_index": np.uint16,
                        "pixel_idx_ptr": np.uint32,
                    }.items():
                        assert np.asarray(table[key]).dtype == np.dtype(dtype)
                    np.testing.assert_array_equal(table["wavelength"], wavelength)
                    assert table["spectral_order"] == order
                    assert np.ndim(table["source_id"]) == 0
                    assert table["source_id"] == source_id
                    assert 0 < len(table["input_xy"]) <= 5
                    assert table["pixel_idx_ptr"][0] == 0
                    assert table["pixel_idx_ptr"][-1] == len(table["fractional_area"])
                    assert len(table["pixel_idx_ptr"]) == len(table["input_xy"]) + 1
                    assert len(table["grism_xy"]) == len(table["wavelength_index"])
                    assert np.all(np.diff(table["pixel_idx_ptr"]) > 0)
                    input_pixels.extend(table["input_xy"][:, 0])
                    expected = disperse.disperse(
                        table["input_xy"][:, 0],
                        table["input_xy"][:, 1],
                        np.full(len(table["input_xy"]), source_id),
                        order,
                        wavelength,
                        observation.direct_image_wcs,
                        observation.grism_wcs,
                        observation.naxis,
                    )
                    for key, values in expected.items():
                        if key != "source_id":
                            np.testing.assert_allclose(table[key], values, rtol=1e-7, atol=0)
            np.testing.assert_array_equal(
                input_pixels, np.where(observation.seg[0] == source_id)[0]
            )

    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(FileExistsError, match="not empty"):
        observation.disperse_order(1, 1.0, 2.0)


def test_file_manifest_no_contributions(tmp_path, monkeypatch):
    observation = make_observation(tmp_path)
    observation.naxis = (1, 1)
    wavelength = np.array([1.0, 2.0])
    monkeypatch.setattr(wfss_geom, "native_wavelength_grid", lambda *args, **kwargs: wavelength)
    manifest = observation.disperse_order(1, 1.0, 2.0, selected_ids=[3])

    assert manifest["sources"] == {3: []}
    assert not list(tmp_path.rglob("*.asdf"))

    empty_manifest = observation.disperse_order(2, 1.0, 2.0, selected_ids=[])
    assert empty_manifest["sources"] == {}


@pytest.mark.parametrize(
    "filename",
    [
        "jw01089001001_03101_00001_nis_cal.fits",
        "jw01089001001_03101_00001_nis_rate.fits",
        "/data/jw01089001001_03101_00001_nis_assignwcsstep.fits",
        "jw01089001001_03101_00001_nis.fits",
    ],
)
def test_default_pdt_directory(filename):
    assert _default_pdt_directory(filename) == Path("jw01089001001_03101_00001_nis_pdt")


@pytest.mark.parametrize("explicit_directory", [None, "custom_pdt"])
def test_step_pdt_directory(tmp_path, monkeypatch, explicit_directory):
    from stdatamodels.jwst import datamodels

    from jwst.wfss_geom.wfss_geom_step import WfssGeomStep

    with datamodels.ImageModel() as model:
        model.meta.filename = "jw01089001001_03101_00001_nis_cal.fits"
        step = WfssGeomStep(output_dir=str(tmp_path), pdt_directory=explicit_directory)
        monkeypatch.setattr(step, "prepare_output", lambda *args, **kwargs: model)
        monkeypatch.setattr(step, "get_reference_file", lambda *args: "reference.fits")
        monkeypatch.setattr(datamodels, "WavelengthrangeModel", lambda *args: nullcontext(None))
        monkeypatch.setattr(datamodels, "open", lambda *args: nullcontext(None))
        directories = []

        def capture_directory(*args, **kwargs):
            directories.append(kwargs["pdt_directory"])
            return {}

        monkeypatch.setattr(wfss_geom, "disperse_all", capture_directory)
        assert step.process(model) == {}
        expected = (
            Path(explicit_directory)
            if explicit_directory is not None
            else tmp_path / "jw01089001001_03101_00001_nis_pdt"
        )
        assert Path(directories[0]) == expected


def test_pdt_directory_overwrite(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        wfss_geom, "native_wavelength_grid", lambda *args, **kwargs: np.array([1.0, 2.0])
    )
    first = make_observation(tmp_path).disperse_order(1, 1.0, 2.0, selected_ids=[1])
    stale_file = Path(first["directory"]) / "stale.asdf"
    stale_file.touch()
    second = make_observation(tmp_path).disperse_order(1, 1.0, 2.0, selected_ids=[2])

    assert first["directory"] == second["directory"] == str((tmp_path / "test_pdt").resolve())
    assert not stale_file.exists()
    assert not Path(first["sources"][1][0]).exists()
    assert Path(second["sources"][2][0]).exists()
    assert "Overwriting existing PDT directory" in caplog.text


def test_failed_pdt_write_removes_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wfss_geom, "native_wavelength_grid", lambda *args, **kwargs: np.array([1.0, 2.0])
    )

    def fail_write(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(wfss_geom.asdf.AsdfFile, "write_to", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        make_observation(tmp_path).disperse_order(1, 1.0, 2.0, selected_ids=[1])

    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.asdf"))


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
