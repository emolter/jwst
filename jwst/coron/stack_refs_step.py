from jwst.datamodels import ModelLibrary

from ..stpipe import Step

from . import stack_refs

__all__ = ["StackRefsStep"]


class StackRefsStep(Step):

    """
    StackRefsStep: Stack multiple PSF reference exposures into a
    single CubeModel, for use by subsequent coronagraphic steps.
    """

    class_alias = "stack_refs"

    spec = """
    """

    def process(self, input):
        """
        input: str or datamodels.ModelLibrary
        """

        if not isinstance(input, ModelLibrary):
            input = ModelLibrary(input)

        # Call the stacking routine
        output_model = stack_refs.make_cube(input)
        output_model.meta.cal_step.stack_psfs = 'COMPLETE'

        return output_model
