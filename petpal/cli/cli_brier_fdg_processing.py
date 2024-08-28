import argparse
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


def read_plasma_glucose_concentration(file_path) -> float:
    return float(np.loadtxt(file_path))


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
    image_io.safe_copy_meta(input_image_path=patlak_ki_image_path,
                            out_image_path=output_file_path)


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
                               run_cmrglc: bool = False,
                               verbose: bool = False):
    
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
        
    out_dir_preproc = os.path.join(out_dir, "preproc")
    out_dir_kinetic_modeling = os.path.join(out_dir, "kinetic_modeling")
    os.makedirs(out_dir_preproc, exist_ok=True)
    os.makedirs(out_dir_kinetic_modeling, exist_ok=True)
    
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
    resample_tac_path = os.path.join(out_dir_preproc, f"{sub_ses_prefix}_desc-onscannertimes_blood.tsv")
   
    out_mod = 'pet'
    lin_fit_thresh_in_mins = 30.0
    cmrlgc_lumped_const = 0.65
    cmrlgc_rescaling_const = 100.0
    
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
        'Verbose': verbose,
        }
    
    sub_preproc = preproc.PreProc(output_directory=out_dir_preproc, output_filename_prefix=sub_ses_prefix)
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
                                                      output_directory=out_dir_kinetic_modeling,
                                                      output_filename_prefix=sub_ses_prefix)
        patlak_obj.run_analysis(method_name='patlak', t_thresh_in_mins=lin_fit_thresh_in_mins)
        patlak_obj.save_analysis()
        
    if run_cmrglc:
        patlak_slope_img = os.path.join(out_dir_kinetic_modeling, f"{sub_ses_prefix}_desc-patlak_slope.nii.gz")
        cmrglc_slope_path = os.path.join(out_dir_kinetic_modeling, f"{sub_ses_prefix}_desc-cmrglc_pet.nii.gz")
        plasma_glc_path   = os.path.join(pet_dir, f"{sub_ses_prefix}_desc-bloodconcentration_glucose.txt")
        
        save_cmrglc_image_from_patlak_ki(patlak_ki_image_path=patlak_slope_img,
                                         output_file_path=cmrglc_slope_path,
                                         plasma_glucose=read_plasma_glucose_concentration(plasma_glc_path),
                                         lumped_constant=cmrlgc_lumped_const,
                                         rescaling_const=cmrlgc_rescaling_const)

_PROG_NAME_ = r"petpal-brier-fdg-pipeline"
_FDG_CMR_EXAMPLE_ = (rf"""
Example:
    - Assuming if we are in the `code` directory of a BIDS directory:
      {_PROG_NAME_} --sub-id XXXX --ses-id XX
""")

def main():
    
    
    parser = argparse.ArgumentParser(prog=f'{_PROG_NAME_}',
                                     description='Command line interface for generating parametric CMRglc images',
                                     epilog=_FDG_CMR_EXAMPLE_,
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-i', '--sub-id', required=True,
                        help='Subject ID assuming sub_XXXX where XXXX is the subject ID.')
    parser.add_argument('-s', '--ses-id', required=True,
                        help='Session ID assuming ses_XX where XX is the session ID.')
    
    parser.add_argument('-b', '--bids-dir', required=False, default=None,
                        help='Base directory for the BIDS-like data for the study. If not set, assumes current working '
                             'directory is the code/ directory of a BIDS-like directory.')
    parser.add_argument('-p', '--pet4d-dir', required=False, default=None,
                        help='Directory where the raw 4D-PET, raw blood tac, and blood glucose files are.')
    parser.add_argument('-a', '--anat-dir', required=False, default=None,
                        help='Directory where the T1-weighted MR image is.')
    parser.add_argument('-o', '--out-dir', required=False, default=None,
                        help='Directory where the outputs of the pipeline will be saved.')
    
    parser.add_argument('-v', '--verbose', required=False, action='store_true', default=False,
                        help='Verbose information for each step in the pipeline.')
    
    
    parser.add_argument('--skip-crop', required=False, action='store_true', default=False,
                        help='Whether to skip the cropping step in the pipeline.')
    parser.add_argument('--skip-wss', required=False, action='store_true', default=False,
                        help='Whether to skip the weighted-series-sum (wss) step in the pipeline.')
    parser.add_argument('--skip-moco', required=False, action='store_true', default=False,
                        help='Whether to skip the moco step in the pipeline.')
    parser.add_argument('--skip-reg', required=False, action='store_true', default=False,
                        help='Whether to skip the registration step in the pipeline.')
    parser.add_argument('--skip-blood-resample', required=False, action='store_true', default=False,
                        help='Whether to skip the blood resample step in the pipeline.')
    parser.add_argument('--skip-patlak', required=False, action='store_true', default=False,
                        help='Whether to skip the patlak analysis step in the pipeline.')
    parser.add_argument('--skip-cmrglc', required=False, action='store_true', default=False,
                        help='Whether to skip the CMRglc analysis step in the pipeline.')

    args = parser.parse_args()
    
    fdg_protocol_with_arterial(sub_id=args.sub_id,
                               ses_id=args.ses_id,
                               bids_root_dir=args.bids_dir,
                               pet_dir_path=args.pet4d_dir,
                               anat_dir_path=args.anat_dir,
                               run_crop=not args.skip_crop,
                               run_wss=not args.skip_wss,
                               run_moco=not args.skip_moco,
                               run_reg=not args.skip_reg,
                               run_resample=not args.skip_blood_resample,
                               run_patlak=not args.skip_patlak,
                               run_cmrglc=not args.skip_cmrglc,
                               verbose=args.verbose)