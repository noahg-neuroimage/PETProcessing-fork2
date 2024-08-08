# pylint: skip-file
import os
import argparse
import pandas as pd
from petpal.preproc import preproc

_VAT_EXAMPLE_ = (r"""
Example:
  - Running one subject:
    vat-cli --sub sub-001_ses-01 --out-dir /path/to/output --pet /path/to/pet.nii --fs-dir /path/to/subject/FreeSurfer/
  - Running all subjects:
    vat-cli --participants participants.tsv --out-dir /path/to/output --pet-dir /path/to/pet/folder/ --pet-postfix _pet.nii.gz --fs-dir /path/to/subject/FreeSurfer/
  - Verbose:
    vat-cli -v [sub-command] [arguments]
""")


def vat_protocol(subjstring: str,
                 out_dir: str,
                 pet_dir: str,
                 fs_dir: str,
                 reg_dir: str):
    sub, ses = rename_subs(subjstring)
    preproc_props = {
        'FilePathLabelMap': '/data/jsp/human2/goldmann/dseg.tsv',
        'FilePathAtlas': '/data/petsun43/data1/atlas/MNI152/MNI152_T1_1mm.nii',
        'HalfLife': 6586.2,
        'StartTimeWSS': 1800,
        'EndTimeWSS': 7200,
        'MotionTarget': (0,600),
        'RegPars': {'aff_metric': 'mattes','type_of_transform': 'DenseRigid'},
        'RefRegion': 1,
        'BlurSize': 6,
        'TimeFrameKeyword': 'FrameTimesStart',
        'Verbose': True
    }
    if ses=='':
        out_folder = f'{out_dir}/{sub}'
        out_prefix = f'{sub}_pet'
        preproc_props['FilePathMocoInp'] = f'{pet_dir}/{sub}/pet/{sub}_pet.nii.gz'
        preproc_props['FilePathSeg'] = f'{reg_dir}/{sub}/aparc+aseg.nii'
        preproc_props['FilePathAnat'] = f'{reg_dir}/{sub}/{sub}_mpr.nii'
        preproc_props['FreeSurferSubjectDir'] = f'{fs_dir}/{subjstring}/mri/'
    else:
        out_folder = f'{out_dir}/{sub}_{ses}'
        out_prefix = f'{sub}_{ses}_pet'
        preproc_props['FilePathMocoInp'] = f'{pet_dir}/{sub}/{ses}/pet/{sub}_{ses}_trc-18FVAT_pet.nii.gz'
        preproc_props['FilePathSeg'] = f'{reg_dir}/{sub}_{ses}/aparc+aseg.nii'
        preproc_props['FilePathAnat'] = f'{reg_dir}/{sub}_{ses}/{sub}_{ses}_mpr.nii'
        preproc_props['FreeSurferSubjectDir'] = f'{fs_dir}/{subjstring}_Bay3prisma/mri/'
    sub_vat = preproc.PreProc(
        output_directory=out_folder,
        output_filename_prefix=out_prefix
    )
    real_files = [
        preproc_props['FilePathMocoInp'],
        preproc_props['FilePathSeg'],
        preproc_props['FilePathAnat']
    ]
    for check in real_files:
        if not os.path.exists(check):
            print(f'{check} not found')
            return None
    print(real_files)
    preproc_props['FilePathRegInp'] = sub_vat._generate_outfile_path(method_short='moco')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('motion_corr')
    sub_vat.run_preproc('register_pet')
    sub_vat.run_preproc('vat_wm_ref_region')
    preproc_props['FilePathSeg'] = sub_vat._generate_outfile_path(method_short='wm-merged')
    sub_vat.update_props(preproc_props)
    preproc_props['FilePathTACInput'] = sub_vat._generate_outfile_path(method_short='reg')
    preproc_props['FilePathWSSInput'] = sub_vat._generate_outfile_path(method_short='reg')
    preproc_props['FilePathSUVRInput'] = sub_vat._generate_outfile_path(method_short='wss')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('resample_segmentation')
    preproc_props['FilePathSeg'] = sub_vat._generate_outfile_path(method_short='seg-res')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('write_tacs')
    sub_vat.run_preproc('weighted_series_sum')
    sub_vat.run_preproc('suvr')
    return None


def rename_subs(sub: str):
    """
    Handle converting subject ID to BIDS structure.

    VATDYS0XX -> sub-VATDYS0XX
    PIBXX-YYY_VYrZ -> sub-PIBXXYYY_ses-VYrZ

    returns:
        - subject part string
        - session part string
    """
    if 'VAT' in sub:
        return [f'sub-{sub}', '']
    elif 'PIB' in sub:
        subname, sesname = sub.split('_')
        subname = subname.replace('-','')
        subname = f'sub-{subname}'
        sesname = f'ses-{sesname}'
        return [subname, sesname]


def main():
    """
    VAT command line interface
    """
    parser = argparse.ArgumentParser(prog='vat-cli',
                            description='Command line interface for running VAT processing.',
                            epilog=_VAT_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--participants',required=True,help='Path to participants.tsv')
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        raise SystemExit('Exiting without command')

    subs_sheet = pd.read_csv(args.participants,sep='\t')
    subs = subs_sheet['participant_id']

    for sub in subs:
        vat_protocol(subjstring=sub)
