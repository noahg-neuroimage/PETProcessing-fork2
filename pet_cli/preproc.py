"""
Introduces class :class:`PreProc` which handles preprocessing of PET and other
neuroimaging data for a PET study. Acts as a wrapper for other tools supplied
in `PPM` 
"""
import os
from . import qc_plots
from image_operations_4d import weighted_series_sum, write_tacs, extract_tac_from_nifty_using_mask, resample_segmentation
from register import register_pet
from motion_corr import motion_corr


class PreProc():
    """
    :class:`ImageOps4D` to provide basic implementations of the preprocessing functions in module
    ``image_operations_4d``. Uses a properties dictionary ``preproc_props`` to
    determine the inputs and outputs of preprocessing methods.

    Key methods include:
        - :meth:`update_props`: Update properties dictionary ``preproc_props``
          with new properties.
        - :meth:`run_preproc`: Given a method in ``image_operations_4d``, run the
          provided method with inputs and outputs determined by properties
          dictionary ``preproc_props``.

    Attributes:
        -`output_directory`: Directory in which files are written to.
        -`output_filename_prefix`: Prefix appended to beginning of written
         files.
        -`preproc_props`: Properties dictionary used to set parameters for PET
         preprocessing.

    Example:

    .. code-block:: python
        output_directory = '/path/to/processing'
        output_filename_prefix = 'sub-01'
        sub_01 = pet_cli.image_operations_4d.ImageOps4d(output_directory,output_filename_prefix)
        params = {
            'FilePathWSSInput': '/path/to/pet.nii.gz',
            'FilePathAnat': '/path/to/mri.nii.gz',
            'HalfLife': 1220.04,  # C11 half-life in seconds
            'FilePathRegInp': '/path/to/image/to/be/registered.nii.gz',
            'FilePathMocoInp': '/path/to/image/to/be/motion/corrected.nii.gz',
            'MotionTarget': '/path/to/pet/reference/target.nii.gz',
            'FilePathTACInput': '/path/to/registered/pet.nii.gz',
            'FilePathLabelMap': '/path/to/label/map.tsv',
            'FilePathSeg': '/path/to/segmentation.nii.gz',
            'TimeFrameKeyword': 'FrameTimesStart'  # using start time or midpoint reference time
            'Verbose': True,
        }
        sub_01.update_props(params)
        sub_01.run_preproc('weighted_series_sum')
        sub_01.run_preproc('motion_corr')
        sub_01.run_preproc('register_pet')
        sub_01.run_preproc('write_tacs')


    See Also:
        :class:`ImageIO`

    """
    def __init__(self,
                 output_directory: str,
                 output_filename_prefix: str) -> None:
        self.output_directory = os.path.abspath(output_directory)
        self.output_filename_prefix = output_filename_prefix
        self.preproc_props = self._init_preproc_props()


    @staticmethod
    def _init_preproc_props() -> dict:
        """
        Initializes preproc properties dictionary.

        The available fields in the preproc properties dictionary are described
        as follows:
            * FilePathWSSInput (str): Path to file on which to compute weighted series sum.
            * FilePathMocoInp (str): Path to PET file to be motion corrected.
            * FilePathRegInp (str): Path to PET file to be registered to anatomical data.
            * FilePathAnat (str): Path to anatomical image to which ``FilePathRegInp`` is registered.
            * FilePathTACInput (str): Path to PET file with which TACs are computed.
            * FilePathSeg (str): Path to a segmentation image in anatomical space.
            * FilePathLabelMap (str): Path to a label map file, indexing segmentation values to ROIs.
            * MotionTarget (str | tuple): Target for transformation methods. See :meth:`determine_motion_target`.
            * MocoPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.motion_correction`.
            * RegPars (keyword arguments): Keyword arguments fed into the method call :meth:`ants.registration`.
            * HalfLife (float): Half life of radioisotope.
            * RegionExtract (int): Region index in the segmentation image to extract TAC from, if running TAC on a single ROI.
            * TimeFrameKeyword (str): Keyword in metadata file corresponding to frame timing array to be used in analysis.
            * Verbose (bool): Set to ``True`` to output processing information.

        """
        preproc_props = {'FilePathWSSInput': None,
                 'FilePathMocoInp': None,
                 'FilePathRegInp': None,
                 'FilePathAnat': None,
                 'FilePathTACInput': None,
                 'FilePathSeg': None,
                 'FilePathLabelMap': None,
                 'MotionTarget': None,
                 'MocoPars': None,
                 'RegPars': None,
                 'HalfLife': None,
                 'RegionExtract': None,
                 'TimeFrameKeyword': None,
                 'Verbose': False}
        return preproc_props
    

    def update_props(self,new_preproc_props: dict) -> dict:
        """
        Update the processing properties with items from a new dictionary.

        Args:
            new_preproc_props (dict): Dictionary with updated properties 
                values. Can have any or all fields filled from the available
                list of fields.

        Returns:
            updated_props (dict): The updated ``preproc_props`` dictionary.
        """
        preproc_props = self.preproc_props
        valid_keys = [*preproc_props]
        updated_props = preproc_props.copy()
        keys_to_update = [*new_preproc_props]

        for key in keys_to_update:

            if key not in valid_keys:
                raise ValueError("Invalid preproc property! Expected one of:\n"
                                 f"{valid_keys}.\n Got {key}.")

            updated_props[key] = new_preproc_props[key]

        self.preproc_props = updated_props
        return updated_props


    def _check_method_props_exist(self,
                                  method_name: str) -> None:
        """
        Check if all necessary properties exist in the ``props`` dictionary to
        run the given method.

        Args:
            method_name (str): Name of method to be checked. Must be name of 
                a method in this module.
        """
        preproc_props = self.preproc_props
        existing_keys = [*preproc_props]

        if method_name=='weighted_series_sum':
            required_keys = ['FilePathWSSInput','HalfLife','Verbose']
        elif method_name=='motion_corr':
            required_keys = ['FilePathMocoInp','MotionTarget','Verbose']
        elif method_name=='register_pet':
            required_keys = ['MotionTarget','FilePathRegInp','FilePathAnat','Verbose']
        elif method_name=='resample_segmentation':
            required_keys = ['FilePathTACInput','FilePathSeg','Verbose']
        elif method_name=='extract_tac_from_nifty_using_mask':
            required_keys = ['FilePathTACInput','FilePathSeg','RegionExtract','Verbose']
        elif method_name=='write_tacs':
            required_keys = ['FilePathTACInput','FilePathLabelMap','FilePathSeg','Verbose','TimeFrameKeyword']
        else:
            raise ValueError("Invalid method_name! Must be either"
                             "'weighted_series_sum', 'motion_corr', "
                             "'register_pet', 'resample_segmentation', "
                             "'extract_tac_from_4dnifty_using_mask', or "
                             f"'write_tacs'. Got {method_name}")
        for key in required_keys:
            if key not in existing_keys:
                raise ValueError(f"Preprocessing method requires property"
                                 f" {key}, however {key} was not found in "
                                 "processing properties. Existing properties "
                                 f"are: {existing_keys}, while needed keys to "
                                 f"run {method_name} are: {required_keys}.")


    def _generate_outfile_path(self,
                               method_short: str):
        output_file_name = f'{self.output_filename_prefix}_wss.nii.gz'
        outfile = os.path.join(self.output_directory,
                                   output_file_name)

    def run_preproc(self,
                    method_name: str):
        """
        Run a specific preprocessing step.

        Args:
            method_name (str): Name of method to be run. Must be name of a
                method in this module.
        """
        preproc_props = self.preproc_props
        self._check_method_props_exist(method_name=method_name)

        if method_name=='weighted_series_sum':
            output_file_name = f'{self.output_filename_prefix}_wss.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            weighted_series_sum(input_image_4d_path=preproc_props['FilePathWSSInput'],
                                out_image_path=outfile,
                                half_life=preproc_props['HalfLife'],
                                verbose=preproc_props['Verbose'])

        elif method_name=='motion_corr':
            output_file_name = f'{self.output_filename_prefix}_moco.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            moco_outputs = motion_corr(input_image_4d_path=preproc_props['FilePathMocoInp'],
                                       motion_target_option=preproc_props['MotionTarget'],
                                       out_image_path=outfile,
                                       verbose=preproc_props['Verbose'],
                                       half_life=preproc_props['HalfLife'],
                                       kwargs=preproc_props['MocoPars'])
            motion = moco_outputs[2]
            output_plot = os.path.join(self.output_directory,
                                       f'{self.output_filename_prefix}_motion.png')
            qc_plots.motion_plot(framewise_displacement=motion,
                                 output_plot=output_plot)
            return moco_outputs

        elif method_name=='register_pet':
            output_file_name = f'{self.output_filename_prefix}_reg.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            register_pet(motion_target_option=preproc_props['MotionTarget'],
                         input_reg_image_path=preproc_props['FilePathRegInp'],
                         reference_image_path=preproc_props['FilePathAnat'],
                         out_image_path=outfile,
                         verbose=preproc_props['Verbose'],
                         half_life=preproc_props['HalfLife'],
                         kwargs=preproc_props['RegPars'])

        elif method_name=='resample_segmentation':
            output_file_name = f'{self.output_filename_prefix}_seg-res.nii.gz'
            outfile = os.path.join(self.output_directory,
                                   output_file_name)
            resample_segmentation(input_image_4d_path=preproc_props['FilePathTACInput'],
                                  segmentation_image_path=preproc_props['FilePathSeg'],
                                  out_seg_path=outfile,
                                  verbose=preproc_props['Verbose'])
            self.update_props({'FilePathSeg': outfile})

        elif method_name=='extract_tac_from_4dnifty_using_mask':
            return extract_tac_from_nifty_using_mask(input_image_4d_path=preproc_props['FilePathTACInput'],
                                                     segmentation_image_path=preproc_props['FilePathSeg'],
                                                     region=preproc_props['RegionExtract'],
                                                     verbose=preproc_props['Verbose'])

        elif method_name=='write_tacs':
            outdir = os.path.join(self.output_directory,'tacs')
            os.makedirs(outdir,exist_ok=True)
            write_tacs(input_image_4d_path=preproc_props['FilePathTACInput'],
                       label_map_path=preproc_props['FilePathLabelMap'],
                       segmentation_image_path=preproc_props['FilePathSeg'],
                       out_tac_dir=outdir,
                       verbose=preproc_props['Verbose'],
                       time_frame_keyword=preproc_props['TimeFrameKeyword'])

        else:
            raise ValueError("Invalid method_name! Must be either"
                             "'weighted_series_sum', 'motion_corr', "
                             "'register_pet', 'resample_segmentation', "
                             "'extract_tac_from_4dnifty_using_mask', or "
                             f"'write_tacs'. Got {method_name}")

        return None
