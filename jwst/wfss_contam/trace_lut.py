"""Grid-based lookup table that speeds up the grism trace shape computation."""

import numpy as np
from astropy.modeling.mappings import Mapping
from scipy.interpolate import RegularGridInterpolator

__all__ = ["TraceLUT", "build_trace_lut"]


class TraceLUT:
    """
    Interpolated replacement for the "detector" to "grism_detector" WCS transform.

    Caches the ``(x0, y0, wavelength) -> (dx, dy)`` dispersed-pixel offset on a
    coarse regular grid (built once per spectral order using the exact transform),
    then linearly interpolates to approximate the mapping for the many actual
    direct-image pixels that need to be dispersed. This avoids repeating the
    expensive per-pixel wavelength inversion since the trace shape varies smoothly
    across the detector.
    """

    def __init__(self, x0_grid, y0_grid, lam_grid, dx_grid, dy_grid):
        """
        Initialize the lookup table from precomputed grid offsets.

        Parameters
        ----------
        x0_grid, y0_grid : np.ndarray
            1-D, strictly increasing arrays of detector x, y grid positions.
        lam_grid : np.ndarray
            1-D, strictly increasing array of wavelength grid positions, in microns.
        dx_grid, dy_grid : np.ndarray
            Arrays of shape ``(len(x0_grid), len(y0_grid), len(lam_grid))`` giving the
            dispersed-pixel offset from ``(x0, y0)`` at each grid point.
        """
        self._x0_grid = np.asarray(x0_grid)
        self._y0_grid = np.asarray(y0_grid)
        self._lam_grid = np.asarray(lam_grid)
        self._dx_grid = np.asarray(dx_grid)
        self._dy_grid = np.asarray(dy_grid)

        # Stack dx, dy into a single vector-valued interpolator so the grid-cell
        # lookup (the dominant cost) is only done once per query point instead of
        # twice. Used by the generic, elementwise __call__ below.
        values = np.stack([self._dx_grid, self._dy_grid], axis=-1)
        self._interp = RegularGridInterpolator(
            (self._x0_grid, self._y0_grid, self._lam_grid),
            values,
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, x0, y0, wavelength):
        """
        Interpolate the dispersed pixel position for input pixel position(s) and wavelength(s).

        This generic, elementwise path handles arbitrary (independently varying)
        input shapes. For the common case where many pixels share the same
        wavelength array (as in `~jwst.wfss_contam.disperse._disperse_onto_grism`),
        use `evaluate_grid` instead, which is substantially faster.

        Parameters
        ----------
        x0, y0 : np.ndarray
            Detector x, y position(s) of the input (direct image) pixel(s).
        wavelength : np.ndarray
            Wavelength(s), in microns, at which to evaluate the dispersion.
            Must have the same shape as ``x0`` and ``y0``.

        Returns
        -------
        x, y : np.ndarray
            Interpolated x, y position(s) in the dispersed (grism) image, same shape as input.
        """
        x0 = np.asarray(x0)
        y0 = np.asarray(y0)
        shape = x0.shape
        pts = np.stack([x0.ravel(), y0.ravel(), np.asarray(wavelength).ravel()], axis=-1)
        offsets = self._interp(pts).reshape(*shape, 2)
        return x0 + offsets[..., 0], y0 + offsets[..., 1]

    def evaluate_grid(self, x0, y0, wavelength):
        """
        Efficiently evaluate the LUT on the outer-product grid used by ``disperse()``.

        Every pixel is dispersed at the same set of wavelengths, so this exploits the
        separability of trilinear interpolation: the ``(x0, y0)`` grid-cell lookup is
        done once per pixel (independent of wavelength), and the wavelength grid-cell
        lookup is done once for the shared wavelength array (independent of pixel).
        This avoids the redundant ``O(n_pixels * n_wavelengths)`` grid-cell lookups
        that a generic n-dimensional interpolator (e.g. `__call__`) would otherwise
        perform for every one of the many repeated ``(x0, y0)`` / wavelength
        combinations, which dominates the cost of a naive implementation.

        Parameters
        ----------
        x0, y0 : np.ndarray
            1-D arrays of shape ``(n_pixels,)`` giving the detector x, y position of
            each pixel to disperse.
        wavelength : np.ndarray
            1-D array of shape ``(n_wave,)`` giving the wavelengths to evaluate,
            shared by every pixel.

        Returns
        -------
        x, y : np.ndarray
            Arrays of shape ``(n_wave, n_pixels)`` giving the interpolated dispersed
            pixel positions.
        """
        x0 = np.asarray(x0)
        y0 = np.asarray(y0)
        wavelength = np.asarray(wavelength)

        ix, wx = self._cell_weights(self._x0_grid, x0)
        iy, wy = self._cell_weights(self._y0_grid, y0)
        ilam, wlam = self._cell_weights(self._lam_grid, wavelength)

        dx = self._separable_interp(self._dx_grid, ix, wx, iy, wy, ilam, wlam)
        dy = self._separable_interp(self._dy_grid, ix, wx, iy, wy, ilam, wlam)
        return x0[np.newaxis, :] + dx, y0[np.newaxis, :] + dy

    @staticmethod
    def _cell_weights(grid, values):
        """
        Return the lower grid index and fractional weight for linear interpolation.

        Parameters
        ----------
        grid : np.ndarray
            1-D, strictly increasing array of grid points.
        values : np.ndarray
            Values to locate within the grid.

        Returns
        -------
        idx : np.ndarray
            Indices of the lower grid points for each value.
        w : np.ndarray
            Fractional weights for linear interpolation between the lower and upper grid points.
        """
        idx = np.clip(np.searchsorted(grid, values) - 1, 0, len(grid) - 2)
        w = (values - grid[idx]) / (grid[idx + 1] - grid[idx])
        return idx, w

    @staticmethod
    def _separable_interp(grid_vals, ix, wx, iy, wy, ilam, wlam):
        """
        Bilinear-in-(x0, y0), then linear-in-wavelength interpolation of ``grid_vals``.

        Parameters
        ----------
        grid_vals : np.ndarray
            3-D array of shape ``(len(x0_grid), len(y0_grid), len(lam_grid))``
            containing the values to interpolate.
        ix, wx : np.ndarray
            Lower indices and fractional weights for the x0 grid.
        iy, wy : np.ndarray
            Lower indices and fractional weights for the y0 grid.
        ilam, wlam : np.ndarray
            Lower indices and fractional weights for the wavelength grid.

        Returns
        -------
        interp_vals : np.ndarray
            Interpolated values at the specified pixel and wavelength coordinates.
        """
        # Bilinear interpolation over x0, y0 across the full wavelength grid axis,
        # done once per pixel rather than once per (pixel, wavelength) pair.
        c00 = grid_vals[ix, iy, :]
        c10 = grid_vals[ix + 1, iy, :]
        c01 = grid_vals[ix, iy + 1, :]
        c11 = grid_vals[ix + 1, iy + 1, :]
        curve = (
            c00 * ((1 - wx) * (1 - wy))[:, np.newaxis]
            + c10 * (wx * (1 - wy))[:, np.newaxis]
            + c01 * ((1 - wx) * wy)[:, np.newaxis]
            + c11 * (wx * wy)[:, np.newaxis]
        )  # shape (n_pixels, n_wave_grid)

        # Linear interpolation along wavelength, shared across all pixels.
        lo = curve[:, ilam]
        hi = curve[:, ilam + 1]
        return (lo * (1 - wlam)[np.newaxis, :] + hi * wlam[np.newaxis, :]).T


def _native_wavelength_grid(imgxy_to_grismxy, order, wmin, wmax, x_ref, y_ref, oversample_factor=1):
    """
    Determine an appropriate wavelength grid for the trace LUT.

    Uses the same native-dispersion-scale logic as
    `~jwst.wfss_contam.disperse._determine_native_wl_spacing`, evaluated directly at a
    single representative detector position (the native spacing is known to vary by only
    a few percent across the detector, so any single reference position is adequate).
    This is intentionally independent of the spatial grid density (``n_grid``), since the
    ``(x0, y0, wavelength) -> (dx, dy)`` mapping is smooth in wavelength and does not need
    to be sampled as finely as the wavelength grid used for the flux dispersion itself.

    Parameters
    ----------
    imgxy_to_grismxy : astropy model
        The "detector" to "grism_detector" transform, reduced to 2 (x, y) outputs.
    order : int
        Spectral order number.
    wmin, wmax : float
        Minimum, maximum wavelength for the dispersed spectra of this spectral order.
    x_ref, y_ref : float
        Representative detector x, y position at which to evaluate the native spacing.
    oversample_factor : float, optional
        Factor by which to oversample the native wavelength spacing. A value of 1
        (the default) matches the resolution used elsewhere to determine when the
        dispersed trace shape has been adequately sampled; the LUT does not benefit
        from the additional oversampling typically applied to the flux dispersion.

    Returns
    -------
    np.ndarray
        Wavelength grid spanning ``[wmin, wmax]``.
    """
    x_ref = np.atleast_1d(x_ref)
    y_ref = np.atleast_1d(y_ref)
    xwmin, ywmin = imgxy_to_grismxy(x_ref, y_ref, np.atleast_1d(wmin), order)
    xwmax, ywmax = imgxy_to_grismxy(x_ref, y_ref, np.atleast_1d(wmax), order)
    dxw = xwmax - xwmin
    dyw = ywmax - ywmin
    dlam = np.abs((wmax - wmin) / (dyw - dxw))[0] / oversample_factor
    npts = max(int(np.ceil((wmax - wmin) / dlam)), 3)
    return np.linspace(wmin, wmax, npts)


def build_trace_lut(grism_wcs, order, wmin, wmax, naxis, n_grid, wave_oversample_factor=1):
    """
    Build a `TraceLUT` for one spectral order by sampling the exact transform on a coarse grid.

    Parameters
    ----------
    grism_wcs : `~gwcs.wcs.WCS`
        The grism WCS object, from which the "detector" to "grism_detector"
        transform is retrieved and evaluated on the coarse grid.
    order : int
        Spectral order number.
    wmin, wmax : float
        Minimum, maximum wavelength for the dispersed spectra of this spectral order.
    naxis : tuple of int
        ``(nx, ny)`` dimensions of the direct image / segmentation map, used to set
        the spatial extent of the grid.
    n_grid : int
        Number of grid points to sample along each of the x and y axes. Must be >= 2.
        The wavelength axis is sampled independently (see ``wave_oversample_factor``),
        since the trace-shape mapping does not vary on the same scale in wavelength
        as it does spatially.
    wave_oversample_factor : float, optional
        Factor by which to oversample the native wavelength spacing (see
        `_native_wavelength_grid`) when building the wavelength axis of the grid.
        Defaults to 1, i.e. the native spacing.

    Returns
    -------
    TraceLUT
        Lookup table object that can be called as ``trace_lut(x0, y0, wavelength)``
        to approximate the exact "detector" to "grism_detector" transform.
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2 to build a trace LUT, got {n_grid}")

    imgxy_to_grismxy = grism_wcs.get_transform("detector", "grism_detector")
    # We only need the x,y outputs, same as in disperse(). Making the number of
    # outputs dynamic handles legacy WCS objects that did not pass x0, y0, and
    # order through the transform unmodified like the current version does.
    n_outputs = len(imgxy_to_grismxy.outputs)
    imgxy_to_grismxy = imgxy_to_grismxy | Mapping((0, 1), n_inputs=n_outputs)

    nx, ny = naxis
    x0_grid = np.linspace(0, nx - 1, n_grid)
    y0_grid = np.linspace(0, ny - 1, n_grid)
    lam_grid = _native_wavelength_grid(
        imgxy_to_grismxy,
        order,
        wmin,
        wmax,
        (nx - 1) / 2.0,
        (ny - 1) / 2.0,
        oversample_factor=wave_oversample_factor,
    )

    xx0, yy0 = np.meshgrid(x0_grid, y0_grid, indexing="ij")
    x0_flat = xx0.ravel()
    y0_flat = yy0.ravel()
    n_pix = x0_flat.size
    n_wave = lam_grid.size

    # Match the (n_wave, n_pixels) calling convention used elsewhere in this
    # subpackage (see disperse._disperse_onto_grism), where x0/y0 are constant
    # along axis 0 and wavelength is constant along axis 1. Some backward grism
    # dispersion transforms rely on this specific broadcasting pattern internally
    # to efficiently invert the wavelength solution for many pixels at once.
    x0_rep = np.repeat(x0_flat[np.newaxis, :], n_wave, axis=0)
    y0_rep = np.repeat(y0_flat[np.newaxis, :], n_wave, axis=0)
    lam_rep = np.repeat(lam_grid[:, np.newaxis], n_pix, axis=1)

    xd, yd = imgxy_to_grismxy(x0_rep, y0_rep, lam_rep, order)

    dx_grid = (xd - x0_rep).reshape(n_wave, n_grid, n_grid)
    dy_grid = (yd - y0_rep).reshape(n_wave, n_grid, n_grid)
    dx_grid = np.moveaxis(dx_grid, 0, -1)
    dy_grid = np.moveaxis(dy_grid, 0, -1)

    return TraceLUT(x0_grid, y0_grid, lam_grid, dx_grid, dy_grid)
