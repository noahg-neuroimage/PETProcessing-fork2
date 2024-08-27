from tabnanny import verbose

from ..preproc import preproc
import os
import numpy as np
from ..utils import image_io
from ..input_function.blood_input import BloodInputFunction
from ..kinetic_modeling.parametric_images import GraphicalAnalysisParametricImage
import nibabel

def resample_blood_data_on_scanner_times(pet4d_path: str,
                                         raw_blood_tac: str,
                                         lin_fit_thresh_in_mins: float,
                                         out_tac_path: str):
    image_meta_data = image_io.load_metadata_for_nifty_with_same_filename(image_path=pet4d_path)
    frame_times = np.asarray(image_meta_data['FrameReferenceTime']) / 60.0
    blood_times, blood_activity = image_io.load_tac(filename=raw_blood_tac)
    blood_intp = BloodInputFunction(time=blood_times, activity=blood_activity, thresh_in_mins=lin_fit_thresh_in_mins)
    resampled_blood = blood_intp.calc_blood_input_function(t=frame_times)
    resampled_tac = np.asarray([frame_times, resampled_blood], dtype=float)
    
    np.savetxt(X=resampled_tac.T, fname=out_tac_path)
    
    return None


def save_cmrglc_image_from_patlak_ki(patlak_ki_image_path: str,
                                     output_file_path: str,
                                     plasma_glucose: float,
                                     lumped_constant: float,
                                     rescaling_const: float):
    patlak_image = image_io.ImageIO(verbose=False).load_nii(image_path=patlak_ki_image_path)
    patlak_affine = patlak_image.affine
    cmr_vals = (plasma_glucose / lumped_constant) * patlak_image.get_fdata() * rescaling_const
    cmr_image = nibabel.Nifti1Image(dataobj=cmr_vals, affine=patlak_affine)
    nibabel.save(cmr_image, f"{output_file_path}")


def fdg_protocol_with_arterial(sub_id: str,
                               ses_id: str,
                               bids_root_dir: str = None,
                               pet_dir_path: str = None,
                               anat_dir_path: str = None,
                               out_dir_path: str = None,
                               run_crop: bool = False,
                               run_wss: bool = False,
                               run_moco: bool = False,
                               run_reg: bool = False,
                               run_resample: bool = False,
                               run_patlak: bool = False,
                               run_cmrglc: bool = False, ):
    
    sub_ses_prefix = f'sub-{sub_id}_ses-{ses_id}'
    
    if bids_root_dir is not None:
        BIDS_ROOT_DIR = os.path.abspath(bids_root_dir)
    else: # Assumes that the file is in 'SOME_BIDS_Dir/code'
        BIDS_ROOT_DIR = os.path.abspath("../")
    
    if out_dir_path is not None:
        out_dir = os.path.abspath(out_dir_path)
    else:
        out_dir = os.path.join(BIDS_ROOT_DIR, "derivatives", "petpal", "pipeline_brier_fdg", f"sub-{sub_id}", f"ses-{ses_id}")
        os.makedirs(out_dir, exist_ok=True)
    
    sub_path = os.path.join(f"{BIDS_ROOT_DIR}", f"sub-{sub_id}", f"ses-{ses_id}")

    if pet_dir_path is not None:
        pet_dir = os.path.abspath(pet_dir_path)
    else:
        pet_dir = os.path.join(f"{sub_path}", "pet")
        
    if anat_dir_path is not None:
        anat_dir = os.path.abspath(anat_dir_path)
    else:
        anat_dir = os.path.join(f"{sub_path}", "anat")
    
    raw_pet_img_path = os.path.join(pet_dir, f"{sub_ses_prefix}_pet.nii.gz")
    t1w_reference_img_path = os.path.join(anat_dir, f"{sub_ses_prefix}_MPRAGE.nii.gz")
    raw_blood_tac_path = os.path.join(pet_dir, f"{sub_ses_prefix}_desc-decaycorrected_blood.tsv")
    resample_tac_path = os.path.join(out_dir, f"{sub_ses_prefix}_desc-onscannertimes_blood.tsv")
    out_mod = 'pet'
    lin_fit_thresh_in_mins = 30.0
    
    preproc_props = {
        'FilePathAnat': t1w_reference_img_path,
        'FilePathCropInput': raw_pet_img_path,
        'CropThreshold': 1.0e-2,
        'HalfLife': 6586.2,
        'StartTimeWSS':0,
        'EndTimeWSS':600,
        'MotionTarget': (0, 600),
        'RegPars': {'aff_metric': 'mattes', 'type_of_transform': 'DenseRigid'},
        'TimeFrameKeyword': 'FrameReferenceTime',
        'Verbose': True,
        }
    
    sub_preproc = preproc.PreProc(output_directory=out_dir, output_filename_prefix=sub_ses_prefix)
    preproc_props['FilePathWSSInput'] = sub_preproc._generate_outfile_path(method_short='threshcropped', modality=out_mod)
    preproc_props['FilePathMocoInp'] = preproc_props['FilePathWSSInput']
    preproc_props['MotionTarget'] = sub_preproc._generate_outfile_path(method_short='wss', modality=out_mod)
    preproc_props['FilePathRegInp'] = sub_preproc._generate_outfile_path(method_short='moco', modality=out_mod)
    preproc_props['FilePathTACInput'] = sub_preproc._generate_outfile_path(method_short='reg', modality=out_mod)
    sub_preproc.update_props(preproc_props)
    
    if run_crop:
        sub_preproc.run_preproc(method_name='thresh_crop', modality=out_mod)
    if run_wss:
        sub_preproc.run_preproc(method_name='weighted_series_sum', modality=out_mod)
    if run_moco:
        sub_preproc.run_preproc(method_name='motion_corr', modality=out_mod)
    if run_reg:
        sub_preproc.run_preproc(method_name='register_pet', modality=out_mod)
    if run_resample:
        resample_blood_data_on_scanner_times(pet4d_path=preproc_props['FilePathTACInput'],
                                             raw_blood_tac=raw_blood_tac_path,
                                             lin_fit_thresh_in_mins=lin_fit_thresh_in_mins,
                                             out_tac_path=resample_tac_path)
    if run_patlak:
        patlak_obj = GraphicalAnalysisParametricImage(input_tac_path=resample_tac_path,
                                                      pet4D_img_path=preproc_props['FilePathTACInput'],
                                                      output_directory=out_dir,
                                                      output_filename_prefix=sub_ses_prefix)
        patlak_obj.run_analysis(method_name='patlak', t_thresh_in_mins=lin_fit_thresh_in_mins)
        patlak_obj.save_analysis()
    if run_cmrglc:
    