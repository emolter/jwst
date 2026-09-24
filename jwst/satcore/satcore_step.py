"""Add here."""

from jwst.satcore.satcore import infill_saturated_cores
from jwst.stpipe import Step

__all__ = ["SatCoreStep"]


class SatCoreStep(Step):
    """
    Add here.

    Add here.
    """

    class_alias = "satcore"

    spec = """
    Add here.
    """

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

        # Call the main routine on the output model
        output_model = infill_saturated_cores(output_model)

        # Set the step status in the output model
        output_model.meta.cal_step.satcore = "COMPLETE"

        return output_model
