from ..preproc import preproc
import os




def fdg_protocol(bids_root_dir: str,
                 sub_id: str,
                 ses_id: str,
                 pet_dir: str,
                 anat_dir: str,
                 out_dir: str):
    
    sub_ses_prefix = f'sub-{sub_id}_ses-{ses_id}'
    
    BIDS_ROOT = os.path.abspath(bids_root_dir)
    
    
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