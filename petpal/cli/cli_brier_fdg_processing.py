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
        out_dir = os.path.join(BIDS_ROOT_DIR, "derivatives", f"{sub_id}", f"{ses_id}")
    
    sub_path = os.path.join(f"{BIDS_ROOT_DIR}", f"sub-{sub_id}", f"ses-{ses_id}")

    if pet_dir_path is not None:
        pet_dir = os.path.abspath(pet_dir_path)
    else:
        pet_dir = os.path.join(f"{sub_path}", "pet")
        
    if anat_dir_path is not None:
        anat_dir = os.path.abspath(anat_dir_path)
    else:
        anat_dir = os.path.join(f"{sub_path}", "anat")
    
    
    
    preproc_props = {
        'FilePathLabelMap': '/data/brier/CMMS_BIDS/derivatives/ROI_mask/dseg_forCMMS.tsv',
        'HalfLife': 6586.2,
        'StartTimeWSS':0,
        'EndTimeWSS':600,
        'MotionTarget': (0, 600),
        'RegPars': {'aff_metric': 'mattes', 'type_of_transform': 'DenseRigid'},
        'TimeFrameKeyWord': 'FrameTimeStart',
        'Verbose': True,
        }
    
    sub_preproc = preproc.PreProc(output_directory=out_dir,
                                  output_filename_prefix=f'sub-{sub_id}_ses-{ses_id}')