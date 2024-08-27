from ..preproc import preproc
import os




def fdg_protocol(sub_id: str,
                 ses_id: str,
                 bids_root_dir: str = None,
                 pet_dir_path: str = None,
                 anat_dir_path: str = None,
                 out_dir_path: str = None):
    
    sub_ses_prefix = f'sub-{sub_id}_ses-{ses_id}'
    
    if bids_root_dir is not None:
        BIDS_ROOT_DIR = os.path.abspath(bids_root_dir)
    else: # Assumes that the file is in 'SOME_BIDS_Dir/code'
        BIDS_ROOT_DIR = os.path.abspath("../")
    
    if out_dir_path is not None:
        out_dir = os.path.abspath(out_dir_path)
    else:
        out_dir = os.path.join(BIDS_ROOT_DIR, "derivatives", "petpal", "brier_fdg_pipeline", f"sub-{sub_id}", f"ses-{ses_id}")
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
    
    preproc_props = {
        'FilePathCropInput': raw_pet_img_path,
        'CropThreshold': 1.0e-2,
        'HalfLife': 6586.2,
        'StartTimeWSS':0,
        'EndTimeWSS':600,
        'MotionTarget': (0, 600),
        'RegPars': {'aff_metric': 'mattes', 'type_of_transform': 'DenseRigid'},
        'TimeFrameKeyword': 'FrameTimeStart',
        'Verbose': True,
        }
    
    sub_preproc = preproc.PreProc(output_directory=out_dir,
                                  output_filename_prefix=sub_ses_prefix)
    sub_preproc.update_props(preproc_props)
    
    sub_preproc.run_preproc('thresh_crop')
    # sub_preproc.run_preproc('motion_corr')
    # sub_preproc.run_preproc('register_pet')
    