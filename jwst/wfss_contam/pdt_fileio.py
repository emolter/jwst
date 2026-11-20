import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import asdf
import numpy as np

__all__ = ["write_pdt_to_files", "load_pdt_manifest", "open_pdt_part"]


def _storage_uint(values, dtype):  # numpydoc ignore:RT01
    """Copy an integer array to its on-disk dtype."""
    return values.astype(dtype, copy=True)


def write_pdt_to_files(geometry, directory, source_parts):
    """
    Disperse a single chunk and then write a PDT for each source in the chunk to disk.

    Parameters
    ----------
    geometry : dict
        Geometry dictionary returned by the `disperse` function.
    directory : str or Path
        Directory where the PDT files will be written.
    source_parts : dict
        Dictionary mapping source IDs to their part numbers.

    Returns
    -------
    files : dict
        Source IDs mapped to written part filenames and their input pixel coordinates.
    """
    files = {source_id: [] for source_id in source_parts}

    order = geometry["order"]
    wavelength = geometry["wavelength"]
    source_ids, starts, sizes = np.unique(
        geometry["source_id"], return_index=True, return_counts=True
    )
    for source_id, start, size in zip(source_ids, starts, sizes, strict=True):
        stop = start + size
        pointers = geometry["pixel_idx_ptr"][start : stop + 1]
        first, last = int(pointers[0]), int(pointers[-1])
        if first == last:
            continue
        has_contributions = np.diff(pointers) > 0
        compact_pointers = np.concatenate((pointers[:-1][has_contributions], pointers[-1:])) - first

        filename = Path(directory) / (
            f"source_{source_id:06d}_order_{order}_part_{source_parts[source_id]:04d}.asdf"
        )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as handle:
                temporary_path = Path(handle.name)
                tree = {
                    "source_id": int(source_id),
                    "wavelength": wavelength.astype(np.float32),
                    "spectral_order": int(order),
                    "input_xy": geometry["input_xy"][start:stop][has_contributions],
                    "pixel_idx_ptr": _storage_uint(compact_pointers, np.uint32),
                    "grism_xy": _storage_uint(geometry["grism_xy"][first:last], np.uint16),
                    "wavelength_index": _storage_uint(
                        geometry["wavelength_index"][first:last], np.uint16
                    ),
                    "fractional_area": geometry["fractional_area"][first:last].astype(np.float32),
                }
                with asdf.AsdfFile(tree) as table:
                    # writing checksums is slow
                    table.write_to(handle.file, write_checksums=False)
            temporary_path.replace(filename)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        files[source_id].append({"filename": filename.name, "input_xy": tree["input_xy"].tolist()})

    return files


def load_pdt_manifest(directory, order, selected_ids):
    """
    Read the order manifest without loading the contribution arrays.

    Parameters
    ----------
    directory : str or Path
        Directory containing the completed order manifest and its ASDF files.
    order : int
        Expected spectral order of the cached parts.
    selected_ids : iterable of int
        Sources to load from the cache.

    Returns
    -------
    wavelength : ndarray
        Full-precision order wavelength grid.
    parts : list of tuple
        Source IDs, paths, and input pixel coordinates for the selected parts.
    """
    directory = Path(directory)
    with (directory / "manifest.json").open() as manifest_file:
        manifest = json.load(manifest_file)
    sources = manifest["sources"]
    wavelength = np.asarray(manifest["wavelength"], dtype=float)

    missing = set(map(int, selected_ids)) - {int(source_id) for source_id in sources}
    if missing:
        raise ValueError(
            f"PDT cache is missing selected sources for order {order}: {sorted(missing)}"
        )

    parts = []
    for source_id in sorted(set(map(int, selected_ids))):
        for part in sources[str(source_id)]:
            parts.append((source_id, directory / part["filename"], part["input_xy"]))
    return wavelength, parts


@contextmanager
def open_pdt_part(path, order, source_id):
    """Keep a source part open while its lazily loaded geometry is used."""
    with asdf.open(path, lazy_load=True) as table:
        geometry = table.tree
        if geometry["spectral_order"] != order or geometry["source_id"] != source_id:
            raise ValueError(f"PDT source or spectral order does not match manifest: {path}")
        yield geometry
