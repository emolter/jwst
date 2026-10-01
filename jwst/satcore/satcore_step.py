"""Add here."""

from stdatamodels import filetype
from stdatamodels.jwst.datamodels import ImageModel

from jwst.datamodels import ModelContainer, ModelLibrary
from jwst.satcore.satcore import infill_saturated_cores, make_unique_grids
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
    replace_boxsize = integer(default=10) # Half-width of the box around each fitted star center to replace NaN pixels.
    in_memory = boolean(default=True) # Whether to keep models in memory rather than on disk.
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
        # Make a copy if needed for an input model.
        # Don't open filenames if they're not already models --
        # leave it to the ModelLibrary call below to open them.
        input_model = self.prepare_output(step_input, open_models=False)

        if isinstance(input_model, ModelLibrary):
            # Input is already a library: leave it alone.
            output_models = input_model
        elif isinstance(input_model, ImageModel) or (
            isinstance(input_model, str) and filetype.check(input_model) in ["fits", "asdf"]
        ):
            # Input is a single file: pass it to ModelLibrary in a list
            output_models = ModelLibrary([input_model], on_disk=not self.in_memory)
            self.blendheaders = False
        elif isinstance(input_model, (str, dict, list, ModelContainer)):
            # Input is an association or list of models/files
            output_models = ModelLibrary(input_model, on_disk=not self.in_memory)
        else:
            # Input is not recognized
            raise TypeError(f"Input {step_input} is not a 2D image.")

        # First build PSF grids only for unique instrument/filter/detector
        unique_grids = make_unique_grids(
            output_models,
            oversample=self.oversample,
            num_psfs=self.num_psfs,
            fov_pixels=self.fov_pixels,
        )

        # Use pre-computed PSF grids to infill saturated pixels in cores
        with output_models:
            for model in output_models:
                model = infill_saturated_cores(
                    model,
                    unique_grids,
                    fov_pixels=self.fov_pixels,
                    replace_boxsize=self.replace_boxsize,
                )
                output_models.shelve(model)

        # Set the step status in the output model
        record_step_status(output_models, "satcore", status="COMPLETE")

        return output_models
