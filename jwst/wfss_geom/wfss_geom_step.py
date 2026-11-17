import logging

from stdatamodels.jwst import datamodels

from jwst.stpipe import Step
from jwst.wfss_geom import wfss_geom

__all__ = ["WfssGeomStep"]

log = logging.getLogger(__name__)


class WfssGeomStep(Step):
    """Perform projection of direct image pixels into dispersed plane."""

    class_alias = "wfss_geom"

    spec = """
        maximum_cores = string(default='1')
        orders = list(default=None)  # Spectral orders to process, e.g. 1, or 1,2,3
        magnitude_limit = float(default=None) # Isophotal AB magnitude limit for sources to be included in the contamination correction
        wl_oversample = integer(default=2) # oversampling factor for wavelength grid
        max_pixels_per_chunk = integer(default=5000) # max number of pixels to disperse at once
        pdt_spacing = integer(default=None)  # Spacing of grid points per spatial axis for the trace lookup table; None disables spatial interpolation and uses the full transform at every pixel
        pdt_wl_oversample = float(default=None)  # Oversampling factor for the trace lookup table's wavelength grid; None matches the grid specified by wl_oversample
    """  # noqa: E501

    reference_file_types = ["photom", "wavelengthrange"]

    def process(self, input_data):
        """
        Run the WFSS geometric projection step.

        Parameters
        ----------
        input_data : str or `~stdatamodels.jwst.datamodels.ImageModel`
            The input rate file after running assign_wcs.

        Returns
        -------
        output_model : `~stdatamodels.jwst.datamodels.ImageModel`
            Still TBD what goes here.
        """
        output_model = self.prepare_output(input_data, open_as_type=datamodels.ImageModel)

        # Get the wavelengthrange ref file
        waverange_ref = self.get_reference_file(output_model, "wavelengthrange")
        log.info(f"Using WAVELENGTHRANGE reference file {waverange_ref}")

        # get the photom reference file
        photom_ref = self.get_reference_file(output_model, "photom")
        log.info(f"Using PHOTOM reference file {photom_ref}")

        # Get requested orders
        orders = [int(o) for o in self.orders] if self.orders else None

        with (
            datamodels.WavelengthrangeModel(waverange_ref) as waverange_model,
            datamodels.open(photom_ref) as photom_model,
        ):
            pdt = wfss_geom.disperse_all(
                output_model,
                waverange_model,
                photom_model,
                self.maximum_cores,
                orders=orders,
                magnitude_limit=self.magnitude_limit,
                oversample_factor=self.wl_oversample,
                max_pixels_per_chunk=self.max_pixels_per_chunk,
                pdt_spacing=self.pdt_spacing,
                pdt_wl_oversample=self.pdt_wl_oversample,
            )

        if pdt is not None:
            output_model.meta.cal_step.wfss_geom = "COMPLETE"
        else:
            output_model.meta.cal_step.wfss_geom = "SKIPPED"
        return pdt
