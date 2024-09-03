#!/usr/bin/env python
import os.path as op
import os
import shutil
from ..stpipe import Pipeline

from stdatamodels.jwst import datamodels

from jwst.datamodels import ModelLibrary

from ..model_blender import blendmeta

# step imports
from ..coron import stack_refs_step
from ..coron import align_refs_step
from ..coron import klip_step
from ..outlier_detection import outlier_detection_step
from ..resample import resample_step

__all__ = ['Coron3Pipeline']


def to_list_of_planes(model):
    """Convert to a list of ImageModels for each plane"""

    container = []
    for plane in range(model.shape[0]):
        image = datamodels.ImageModel()
        for attribute in [
                'data', 'dq', 'err', 'zeroframe', 'area',
                'var_poisson', 'var_rnoise', 'var_flat'
        ]:
            try:
                setattr(image, attribute, model.getarray_noinit(attribute)[plane])
            except AttributeError:
                pass
        image.update(model)
        try:
            image.meta.wcs = model.meta.wcs
        except AttributeError:
            pass
        container.append(image)
    return container


class Coron3Pipeline(Pipeline):
    """Class for defining Coron3Pipeline.

    Coron3Pipeline: Apply all level-3 calibration steps to a
    coronagraphic association of exposures. Included steps are:

    #. stack_refs (assemble reference PSF inputs)
    #. align_refs (align reference PSFs to target images)
    #. klip (PSF subtraction using the KLIP algorithm)
    #. outlier_detection (flag outliers)
    #. resample (image combination and resampling)

    """

    class_alias = "calwebb_coron3"

    spec = """
        suffix = string(default='i2d')
        in_memory = boolean(default=False)
    """

    # Define aliases to steps
    step_defs = {
        'stack_refs': stack_refs_step.StackRefsStep,
        'align_refs': align_refs_step.AlignRefsStep,
        'klip': klip_step.KlipStep,
        'outlier_detection': outlier_detection_step.OutlierDetectionStep,
        'resample': resample_step.ResampleStep
    }

    prefetch_references = False

    # Main processing
    def process(self, user_input):
        """Primary method for performing pipeline.

        Parameters
        ----------
        user_input : str, Level3 Association, or ~jwst.datamodels.JwstDataModel
            The exposure or association of exposures to process
        """
        self.log.info('Starting calwebb_coron3 ...')
        asn_exptypes = ['science', 'psf']

        # Create a DM object using the association table
        input_models = ModelLibrary(user_input, asn_exptypes=asn_exptypes, on_disk=not self.in_memory)

        # This asn_id assignment is important as it allows outlier detection
        # to know the asn_id since that step receives the cube as input.
        self.asn_id = input_models.asn["asn_id"]

        # Store the output file for future use
        self.output_file = input_models.asn["products"][0]["name"]

        # Set up required output products and formats
        self.outlier_detection.suffix = 'crfints'
        self.outlier_detection.mode = 'coron'
        self.outlier_detection.save_results = self.save_results
        self.resample.blendheaders = False

        # Save the original outlier_detection.skip setting from the
        # input, because it may get toggled off within loops for
        # processing individual inputs
        skip_outlier_detection = self.outlier_detection.skip

        # Extract lists of all the PSF and science target members
        psf_indices = input_models.indices_for_exptype('psf')
        targ_indices = input_models.indices_for_exptype('science')
        input_filenames = [input_models.asn['products'][0]['members'][i]['expname'] for i in range(len(input_models._members))]
        targ_filenames = [input_models.asn['products'][0]['members'][i]['expname'] for i in targ_indices]

        # Make sure we found some PSF and target members
        if len(psf_indices) == 0:
            err_str1 = 'No reference PSF members found in association table.'
            self.log.error(err_str1)
            self.log.error('Calwebb_coron3 processing will be aborted')
            return

        if len(targ_indices) == 0:
            err_str1 = 'No science target members found in association table'
            self.log.error(err_str1)
            self.log.error('Calwebb_coron3 processing will be aborted')
            return

        for member in input_models.asn["products"][0]["members"]:
            self.prefetch(member["expname"])


        # Perform outlier detection on the PSFs.
        psf_models = [] # FIXME: if possible avoid this, right now not possible because stack_refs makes cubemodel
        if not skip_outlier_detection:
            with input_models:
                for i in psf_indices:
                    model = input_models.borrow(i)
                    self.outlier_detection(model)
                    # step may have been skipped for this model;
                    # turn back on for next model
                    self.outlier_detection.skip = False
                    psf_models.append(model)
                    input_models.shelve(model, i)
                del model
        else:
            self.log.info('Outlier detection skipped for PSF\'s')

        # Stack all the PSF images into a single CubeModel
        # FIXME: this loads all models into memory since it makes the list of models into a single model
        # how many PSF observations would we normally expect? look at large dataset that failed with 1 TB memory
        psf_stack = self.stack_refs(psf_models)
        [model.close() for model in psf_models]
        del psf_models

        # Save the resulting PSF stack
        self.save_model(psf_stack, suffix='psfstack')

        # Call the sequence of steps: outlier_detection, align_refs, and klip
        # once for each input target exposure
        resample_input_files = []
        with input_models:
            for i in targ_indices:
                
                target = input_models.borrow(i)
                target_file = input_filenames[i]
                # Remove outliers from the target
                if not skip_outlier_detection:
                    target = self.outlier_detection(target)
                    # step may have been skipped for this model;
                    # turn back on for next model
                    self.outlier_detection.skip = False

                # Call align_refs
                psf_aligned = self.align_refs(target, psf_stack)

                # Call KLIP
                psf_sub = self.klip(target, psf_aligned)
                del psf_aligned

                # remove the _psfalign library.
                # Future improvement will be to make this an optional output
                tmpdir_psfalign = op.join(os.getcwd(), self.align_refs.output_dir)
                shutil.rmtree(tmpdir_psfalign)

                # Save the psf subtraction results
                self.save_model(
                    psf_sub, output_file=target_file,
                    suffix='psfsub', acid=self.asn_id
                )

                # Split out the integrations into separate ImageModels
                # to pass to `resample`
                for j, model in enumerate(to_list_of_planes(psf_sub)):
                    fname = f"plane{i}_{j}.fits"
                    model.save(fname)
                    resample_input_files.append(fname)

                del psf_sub
                input_models.shelve(target, i, modify=False)
            del target

        resample_input_members = [{'expname': fname, 'exptype': 'science'} for fname in resample_input_files]
        resample_asn = {"products":[{"name":"coron3_resample_input","members":resample_input_members}]}
        resample_library = ModelLibrary(resample_asn, on_disk=not self.in_memory)

        # Call the resample step to combine all psf-subtracted target images
        # Output is a single datamodel
        result = self.resample(resample_library)
        [os.remove(fname) for fname in resample_input_files]

        # Blend the science headers
        try:
            completed = result.meta.cal_step.resample
        except AttributeError:
            self.log.debug('Could not determine if resample was completed.')
            self.log.debug('Presuming not.')

            completed = 'SKIPPED'
        if completed == 'COMPLETE':
            self.log.debug(f'Blending metadata for {result}')
            blendmeta.blendmodels(result, inputs=targ_filenames)

        try:
            result.meta.asn.pool_name = input_models.meta.asn_table.asn_pool
            result.meta.asn.table_name = op.basename(user_input)
        except AttributeError:
            self.log.debug('Cannot set association information on final')
            self.log.debug(f'result {result}')

        # Save the final result
        self.save_model(result, suffix=self.suffix)

        # We're done
        self.log.info('...ending calwebb_coron3')

        return
