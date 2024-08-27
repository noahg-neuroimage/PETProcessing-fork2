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
    preproc_props['FilePathWSSInput'] = sub_preproc._generate_outfile_path(method_short='threshcropped', modality='pet')
    preproc_props['FilePathMocoInp'] = preproc_props['FilePathWSSInput']
    preproc_props['MotionTarget'] = sub_preproc._generate_outfile_path(method_short='wss', modality='pet')
    preproc_props['FilePathRegInp'] = sub_preproc._generate_outfile_path(method_short='moco', modality='pet')
    sub_preproc.update_props(preproc_props)
    
    sub_preproc.run_preproc(method_name='thresh_crop', modality='pet')
    
    sub_preproc.run_preproc(method_name='weighted_series_sum', modality='pet')

    sub_preproc.run_preproc(method_name='motion_corr', modality='pet')
    
    sub_preproc.run_preproc(method_name='register_pet', modality='pet')
    