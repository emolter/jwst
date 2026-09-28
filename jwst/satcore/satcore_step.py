"""Add here."""

from jwst.satcore.satcore import infill_saturated_cores
from jwst.stpipe import Step, record_step_status

__all__ = ["SatCoreStep"]


class SatCoreStep(Step):
    """
    Detect bright stars, model their PSF, and fill in NaNs near their cores.

    This step is used for correcting saturation in the cores of foreground stars,
    for the purpose of getting a better estimate of direct-image flux for WFSS forward modeling.
    """

    class_alias = "satcore"

    spec = """
    oversample = integer(default=3) # PSF oversample factor.
    num_psfs = integer(default=36) # Number of points across detector to make an ePSF. Must be a square number.
    fov_pixels = integer(default=51) # Full-width in detector pixels of the PSF model that is fit.
    box_halfwidth = integer(default=10) # Half-width of the box around each fitted star center to replace NaN pixels.
    """  # noqa: E501

    def process(self, step_input):
        """
        Determine the source type.

        Parameters
        ----------
        step_input : str, add datamodel types here
            Either the path to the file or the science data model to be corrected.

        Returns
        -------
        output_model : add datamodel types here
            Data model with...
        """
        output_model = self.prepare_output(step_input)

        # we need to handle output_model being ModelLibrary. Look at how other image3 steps do it.

        for model in output_model:
            # Call the main routine on the output model
            model = infill_saturated_cores(
                model,
                oversample=self.oversample,
                num_psfs=self.num_psfs,
                fov_pixels=self.fov_pixels,
                box_halfwidth=self.box_halfwidth,
            )

        # Set the step status in the output model
        record_step_status(output_model, "satcore", status="COMPLETE")

        return output_model
