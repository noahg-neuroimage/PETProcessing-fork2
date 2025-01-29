# pylint: skip-file
import os
import argparse
import pandas as pd
import ants
from petpal.kinetic_modeling import parametric_images, fit_tac_with_rtms, graphical_analysis
from petpal.pipelines import pipelines


_VAT_EXAMPLE_ = (r"""
Example:
  - Running many subjects:
    petpal-vat-proc --subjects participants.tsv --out-dir /path/to/output --pet-dir /path/to/pet/folder/ --reg-dir /path/to/subject/Registrations/
""")
def vat_protocol(subjstring: str,
                 out_dir: str,
                 pet_dir: str,
                 reg_dir: str,
                 skip: bool):
    sub, ses = rename_subs(subjstring)
    preproc_props = {
        'FilePathLabelMap': '/home/usr/goldmann/dseg.tsv',
        'FilePathFSLPremat': '',
        'FilePathFSLPostmat': '',
        'HalfLife': 6586.2,
        'StartTimeWSS': 1800,
        'EndTimeWSS': 7200,
        'MotionTarget': (0,600),
        'RegPars': {'aff_metric': 'mattes','type_of_transform': 'DenseRigid'},
        'RefRegion': 1,
        'BlurSize': 4.2,
        'TimeFrameKeyword': 'FrameTimesStart',
        'Verbose': True
    }
    if ses=='':
        out_folder = f'{out_dir}/{sub}'
        out_prefix = f'{sub}'
        preproc_props['FilePathMocoInp'] = f'{pet_dir}/{sub}/pet/{sub}_pet.nii.gz'
        preproc_props['FilePathSeg'] = f'{reg_dir}/{sub}/{sub}_aparc+aseg.nii'
        preproc_props['FilePathBSseg'] = f'{reg_dir}/{sub}/{sub}_brainstem.nii'
        preproc_props['FilePathAnat'] = f'{reg_dir}/{sub}/{sub}_mpr.nii'
        preproc_props['FilePathWarp'] = f'{reg_dir}/{sub}/PRISMA_TRIO_PIB_NL_ANTS_NoT2/{sub}_mpr_to_PRISMA_TRIO_PIB_NL_T1_ANTSwarp.nii.gz'
        brain_mask_path = f'{reg_dir}/{sub}/{sub}_mpr_brain_mask.nii'
    else:
        out_folder = f'{out_dir}/{sub}_{ses}'
        out_prefix = f'{sub}_{ses}'
        preproc_props['FilePathMocoInp'] = f'{pet_dir}/{sub}/{ses}/pet/{sub}_{ses}_trc-18FVAT_pet.nii.gz'
        preproc_props['FilePathSeg'] = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_aparc+aseg.nii'
        preproc_props['FilePathBSseg'] = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_brainstem.nii'
        preproc_props['FilePathAnat'] = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_mpr.nii'
        preproc_props['FilePathWarp'] = f'{reg_dir}/{subjstring}_Bay3prisma/PRISMA_TRIO_PIB_NL_ANTS_NoT2/{subjstring}_Bay3prisma_mpr_to_PRISMA_TRIO_PIB_NL_T1_ANTSwarp.nii.gz'
        brain_mask_path = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_mpr_brain_mask.nii'
    real_files = [
        preproc_props['FilePathMocoInp'],
        preproc_props['FilePathSeg'],
        preproc_props['FilePathAnat'],
        preproc_props['FilePathBSseg'],
        preproc_props['FilePathWarp'],
        brain_mask_path
    ]
    for check in real_files:
        if not os.path.exists(check):
            print(f'{check} not found')
            return None
    print(real_files)


    if ses=='':
        sub_flex = sub
    else:
        sub_flex = f'{sub}_{ses}'


    # preprocessing
    vat_pipeline = pipelines.BIDS_Pipeline(sub_id=sub.replace('sub-',''),
                                           ses_id=ses.replace('ses-',''),
                                           segmentation_img_path=f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_aparc+aseg.nii',
                                           segmentation_label_table_path='/home/usr/goldmann/dseg.tsv',
                                           raw_anat_img_path=f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_mpr.nii',
                                           pipeline_name='VATDYS')
    print(vat_pipeline)
    preproc_props['FilePathRegInp'] = sub_vat.generate_outfile_path(method_short='moco')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('motion_corr')
    sub_vat.run_preproc('register_pet')
    sub_vat.run_preproc('vat_wm_ref_region')
    print('finished wm ref region')

    # write tacs
    preproc_props['FilePathSeg'] = sub_vat.generate_outfile_path(method_short='wm-merged')
    preproc_props['FilePathTACInput'] = sub_vat.generate_outfile_path(method_short='reg')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('write_tacs')

    # run patlak
    tacs = os.path.join(out_dir,sub_flex,'tacs')
    graphical_model = graphical_analysis.MultiTACGraphicalAnalysis(
        input_tac_path=os.path.join(tacs,'WMRef_tac.tsv'),
        roi_tacs_dir=tacs,
        output_directory=os.path.join(out_dir,sub_flex),
        output_filename_prefix=out_prefix,
        method='patlak',
        fit_thresh_in_mins=20
    )
    graphical_model.run_analysis()

    # calculate pars and save results
    region_names = [os.path.basename(result['FilePathTTAC']) for result in graphical_model.analysis_props]
    ki = [result['Slope'] for result in graphical_model.analysis_props]
    ki_pandas = pd.DataFrame(columns=['regions', 'ki_patlak'],data={'regions': region_names, 'ki_patlak': ki})
    ki_pandas.to_csv(os.path.join(out_dir,sub_flex,f'{sub_flex}_ki-patlak.tsv'),sep='\t')
    
    # parametric patlak
    parametric_analysis = parametric_images.GraphicalAnalysisParametricImage(
        input_tac_path=os.path.join(tacs,'WMRef_tac.tsv'),
        pet4D_img_path=sub_vat.generate_outfile_path(method_short='reg'),
        output_directory=os.path.join(out_dir,sub_flex),
        output_filename_prefix=out_prefix,
    )
    parametric_analysis.run_analysis(method_name='patlak',t_thresh_in_mins=20,image_scale=1)
    parametric_analysis.save_analysis()

    # suvr
    preproc_props['FilePathWSSInput'] = sub_vat.generate_outfile_path(method_short='reg')
    preproc_props['FilePathSUVRInput'] = sub_vat.generate_outfile_path(method_short='wss')
    sub_vat.update_props(preproc_props)
    sub_vat.run_preproc('weighted_series_sum')
    sub_vat.run_preproc('suvr')

    # pvc
    segfile_4d = sub_vat.generate_outfile_path(method_short='wm-merged-4d')
    #suvr_file = sub_vat.generate_outfile_path(method_short='suvr')
    #suvr_pvc_file = sub_vat.generate_outfile_path(method_short='desc-suvr_pvc-rbv')
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/pvc_make4d -i {preproc_props['FilePathSeg']} -o {segfile_4d}")
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/petpvc -i {suvr_file} -o {suvr_pvc_file} -m {segfile_4d} -p RBV+VC -x 4.2 -y 4.2 -z 4.2")
    ki_file = sub_vat.generate_outfile_path(method_short='desc-patlak-slope')
    #ki_pvc_file = sub_vat.generate_outfile_path(method_short='desc-ki_pvc-rbv')
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/pvc_make4d -i {preproc_props['FilePathSeg']} -o {segfile_4d}")
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/petpvc -i {ki_file} -o {ki_pvc_file} -m {segfile_4d} -p RBV+VC -x 4.2 -y 4.2 -z 4.2")


    # reg ki to atlas
    ref = ants.image_read('/data/jsp/human2/AaronProjects/PRISMA_TRIO_PIB_NL/PRISMA_TRIO_PIB_NL_MNI152_T1_0p9mm.nii.gz')
    ki_ants = ants.image_read(ki_file)
    ki_warp = ants.apply_transforms(
        fixed=ref,
        moving=ki_ants,
        transformlist=[preproc_props['FilePathWarp']],
        imagetype=0,
        verbose=1
    )
    ki_smooth = ants.smooth_image(image=ki_warp,sigma=4.2,FWHM=True)
    ants.image_write(ki_smooth,sub_vat.generate_outfile_path(method_short='desc-ki_space-atlas'))

    #suvr_ants = ants.image_read(suvr_pvc_file)
    #suvr_warp = ants.apply_transforms(
    #    fixed=ref,
    #    moving=suvr_ants,
    #    transformlist=[preproc_props['FilePathWarp']],
    #    imagetype=0,
    #    verbose=1
    #)
    #suvr_smooth = ants.smooth_image(image=suvr_warp,sigma=6,FWHM=True)
    #ants.image_write(suvr_smooth,sub_vat.generate_outfile_path(method_short='desc-suvr_pvc-rbv_space-atlas'))

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
    parser = argparse.ArgumentParser(prog='petpal-vat-proc',
                                     description='Command line interface for running VAT processing.',
                                     epilog=_VAT_EXAMPLE_, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s','--subjects',required=True,help='Path to participants.tsv')
    parser.add_argument('-o','--out-dir',required=True,help='Output directory analyses are saved to.')
    parser.add_argument('-p','--pet-dir',required=True,help='Path to parent directory of PET imaging data.')
    parser.add_argument('-r','--reg-dir',required=True,help='Path to parent directory of registrations computed from MPR to atlas space.')
    args = parser.parse_args()


    subs_sheet = pd.read_csv(args.subjects,sep='\t')
    subs = subs_sheet['participant_id']

    for sub in subs[:1]:
        if sub[:6]=='VATDYS':
            pet_dir = '/data/norris/data1/data_archive/VATDYS'
            reg_dir = '/export/scratch1/Registration/VATDYS'
        elif sub[:5]=='VATNL':
            pet_dir = '/data/norris/data1/data_archive/VATNL'
            reg_dir = '/data/norris/data1/Registration/VATNL'
        elif sub[:3]=='PIB':
            pet_dir = '/data/jsp/human2/BidsDataset/PIB'
            reg_dir = '/data/jsp/human2/Registration/PIB_FS73_ANTS'
        vat_protocol(sub,args.out_dir,pet_dir,reg_dir,skip=True)
        #try:
        #    vat_protocol(sub,args.out_dir,pet_dir,reg_dir,skip=True)
        #except:
        #    print(f"Couldn't run {sub}, passing")
        #    pass

main()
