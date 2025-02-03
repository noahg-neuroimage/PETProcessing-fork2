# pylint: skip-file
import os
import argparse
import pandas as pd
import ants
import nibabel
from petpal.kinetic_modeling import parametric_images, graphical_analysis,rtm_analysis
from petpal.pipelines import pipelines, steps_base, preproc_steps
from petpal.preproc import image_operations_4d, motion_corr, register, segmentation_tools
from petpal.preproc import symmetric_geometric_transfer_matrix as sgtm
from petpal.utils.bids_utils import gen_bids_like_dir_path, gen_bids_like_filename, gen_bids_like_filepath
from petpal.utils.image_io import _HALFLIVES_, km_regional_fits_to_tsv
from petpal.utils import useful_functions


_VAT_EXAMPLE_ = (r"""
Example:
  - Running many subjects:
    petpal-vat-proc --subjects participants.tsv --out-dir /path/to/output --pet-dir /path/to/pet/folder/ --reg-dir /path/to/subject/Registrations/
""")
def vat_protocol(subjstring: str,
                 out_dir: str,
                 pet_dir: str,
                 reg_dir: str,
                 skip: list):
    sub, ses = rename_subs(subjstring)
    sub_id = sub.replace('sub-','')
    ses_id = ses.replace('ses-','')
    segmentation_label_file = '/home/usr/goldmann/dseg.tsv'
    motion_target = (0,600)
    reg_pars = {'aff_metric': 'mattes','type_of_transform': 'DenseRigid'}
    half_life = _HALFLIVES_['f18']
    suvr_start = 1800
    suvr_end = 7200
    preproc_props = {
        'FilePathFSLPremat': '',
        'FilePathFSLPostmat': '',
        'StartTimeWSS': 1800,
        'EndTimeWSS': 7200,
        'RefRegion': 1,
        'BlurSize': 4.2,
        'TimeFrameKeyword': 'FrameTimesStart',
        'Verbose': True
    }
    if ses=='':
        out_folder = f'{out_dir}/{sub}'
        out_prefix = f'{sub}'
        pet_file = f'{pet_dir}/{sub}/pet/{sub}_pet.nii.gz'
        freesurfer_file = f'{reg_dir}/{sub}/{sub}_aparc+aseg.nii'
        brainstem_segmentation_file = f'{reg_dir}/{sub}/{sub}_brainstem.nii'
        mprage_file = f'{reg_dir}/{sub}/{sub}_mpr.nii'
        atlas_warp_file = f'{reg_dir}/{sub}/PRISMA_TRIO_PIB_NL_ANTS_NoT2/{sub}_mpr_to_PRISMA_TRIO_PIB_NL_T1_ANTSwarp.nii.gz'
        mpr_brain_mask_file = f'{reg_dir}/{sub}/{sub}_mpr_brain_mask.nii'
    else:
        out_folder = f'{out_dir}/{sub}_{ses}'
        out_prefix = f'{sub}_{ses}'
        pet_file = f'{pet_dir}/{sub}/{ses}/pet/{sub}_{ses}_trc-18FVAT_pet.nii.gz'
        freesurfer_file = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_aparc+aseg.nii'
        brainstem_segmentation_file = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_brainstem.nii'
        mprage_file = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_mpr.nii'
        atlas_warp_file = f'{reg_dir}/{subjstring}_Bay3prisma/PRISMA_TRIO_PIB_NL_ANTS_NoT2/{subjstring}_Bay3prisma_mpr_to_PRISMA_TRIO_PIB_NL_T1_ANTSwarp.nii.gz'
        mpr_brain_mask_file = f'{reg_dir}/{subjstring}_Bay3prisma/{subjstring}_Bay3prisma_mpr_brain_mask.nii'
    real_files = [
        pet_file,
        freesurfer_file,
        mprage_file,
        brainstem_segmentation_file,
        atlas_warp_file,
        mpr_brain_mask_file
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


    def vat_bids_filepath(suffix,folder,**extra_desc):

        new_dir = gen_bids_like_dir_path(sub_id=sub_id,
                                         ses_id=ses_id,
                                         sup_dir=out_dir,
                                         modality=folder)
        os.makedirs(new_dir,exist_ok=True)
        new_file = gen_bids_like_filepath(sub_id=sub_id,
                                          ses_id=ses_id,
                                          bids_dir=out_dir,
                                          modality=folder,
                                          suffix=suffix,
                                          **extra_desc)

        return new_file


    # preprocessing
    pet_cropped_file = vat_bids_filepath(suffix='pet',folder='pet',crop='003')
    if 'crop' not in skip:
        image_operations_4d.SimpleAutoImageCropper(input_image_path=pet_file,
                                                out_image_path=pet_cropped_file,
                                                thresh_val=0.03)

    pet_moco_file = vat_bids_filepath(suffix='pet',folder='pet',moco='windowed')
    if 'moco' not in skip:
        motion_corr.windowed_motion_corr_to_target(input_image_path=pet_cropped_file,
                                                out_image_path=pet_moco_file,
                                                motion_target_option=motion_target,
                                                w_size=300)

    pet_reg_anat_file = vat_bids_filepath(suffix='pet',folder='pet',moco='windowed',space='mpr')
    if 'register' not in skip:
        register.register_pet(input_reg_image_path=pet_moco_file,
                            out_image_path=pet_reg_anat_file,
                            reference_image_path=mprage_file,
                            motion_target_option=motion_target,
                            half_life=half_life,
                            verbose=True,
                            **reg_pars)
    
    vat_wm_ref_region_roi_file = vat_bids_filepath(suffix='seg',folder='pet',desc='RefRegionROI')
    vat_wm_ref_segmentation_file = vat_bids_filepath(suffix='seg',folder='pet',desc='RefRegionSegmentation')
    if 'refregion' not in skip:
        segmentation_tools.vat_wm_ref_region(input_segmentation_path=freesurfer_file,
                                            out_segmentation_path=vat_wm_ref_region_roi_file)
        segmentation_tools.vat_wm_region_merge(wmparc_segmentation_path=freesurfer_file,
                                            bs_segmentation_path=brainstem_segmentation_file,
                                            wm_ref_segmentation_path=vat_wm_ref_region_roi_file,
                                            out_image_path=vat_wm_ref_segmentation_file)

    tac_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='tacs')
    os.makedirs(tac_save_dir,exist_ok=True)
    if 'tacs' not in skip:
        image_operations_4d.write_tacs(input_image_path=pet_reg_anat_file,
                                    label_map_path=segmentation_label_file,
                                    segmentation_image_path=vat_wm_ref_segmentation_file,
                                    out_tac_dir=tac_save_dir,
                                    verbose=True,
                                    out_tac_prefix=out_prefix,
                                    time_frame_keyword='FrameTimesStart')

    # kinetic modeling
    wmref_tac_path = vat_bids_filepath(suffix='tac',folder='tacs',seg='WMRef',ext='.tsv')
    mrtm_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='mrtm_fits')
    km_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='km')
    os.makedirs(km_save_dir,exist_ok=True)
    mrtm_save_path = vat_bids_filepath(suffix='fits',model='mrtm1',folder='km',ext='.tsv')
    os.makedirs(mrtm_save_dir,exist_ok=True)
    mrtm1_path = gen_bids_like_filename(sub_id=sub_id,ses_id=ses_id,model='mrtm1',suffix='fits',ext='')
    if 'mrtm1' not in skip:
        mrtm1_analysis = rtm_analysis.MultiTACRTMAnalysis(ref_tac_path=wmref_tac_path,
                                                        roi_tacs_dir=tac_save_dir,
                                                        output_directory=mrtm_save_dir,
                                                        output_filename_prefix=mrtm1_path,
                                                        method='mrtm')
        mrtm1_analysis.run_analysis(t_thresh_in_mins=10)
        mrtm1_analysis.save_analysis()
        km_regional_fits_to_tsv(fit_results_dir=mrtm_save_dir,out_tsv_dir=mrtm_save_path)

    logan_save_dir = gen_bids_like_dir_path(sub_id=sub_id,ses_id=ses_id,sup_dir=out_dir,modality='logan_fits')
    logan_save_path = vat_bids_filepath(suffix='fits',model='altlogan',folder='km',ext='.tsv')
    os.makedirs(logan_save_dir,exist_ok=True)
    logan_path = gen_bids_like_filename(sub_id=sub_id,ses_id=ses_id,model='altlogan',suffix='fits',ext='')
    if 'logan' not in skip:
        graphical_model = graphical_analysis.MultiTACGraphicalAnalysis(
            input_tac_path=wmref_tac_path,
            roi_tacs_dir=tac_save_dir,
            output_directory=logan_save_dir,
            output_filename_prefix=logan_path,
            method='alt_logan',
            fit_thresh_in_mins=10
        )
        graphical_model.run_analysis()
        graphical_model.save_analysis()
        km_regional_fits_to_tsv(fit_results_dir=logan_save_dir,out_tsv_dir=logan_save_path)

    # suvr
    wss_file_path = vat_bids_filepath(suffix='pet',folder='pet',space='mpr',desc='WSS')
    suvr_file_path = vat_bids_filepath(suffix='pet',folder='pet',space='mpr',desc='SUVR')
    if 'suvr' not in skip:
        useful_functions.weighted_series_sum(input_image_4d_path=pet_reg_anat_file,
                                            half_life=half_life,
                                            verbose=True,
                                            start_time=suvr_start,
                                            end_time=suvr_end,
                                            out_image_path=wss_file_path)
        image_operations_4d.suvr(input_image_path=wss_file_path,
                                segmentation_image_path=vat_wm_ref_segmentation_file,
                                ref_region=1,
                                out_image_path=suvr_file_path,
                                verbose=True)

    if 'pvc' not in skip:
        suvr_pvc_path = vat_bids_filepath(suffix='pet',folder='pet',space='mpr',pvc='SGTM',desc='SUVR')
        suvr_nibabel = nibabel.load(suvr_file_path)
        wm_ref_segmentation_nibabel = nibabel.load(vat_wm_ref_segmentation_file)
        sgtm_results = sgtm.sgtm(pet_nifti=suvr_nibabel,
                                roi_nifti=wm_ref_segmentation_nibabel,
                                fwhm=4.2,)
        print(sgtm_results[0])
        print(sgtm_results[1])
        print(sgtm_results[2])












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
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/pvc_make4d -i {freesurfer_file} -o {segfile_4d}")
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/petpvc -i {suvr_file} -o {suvr_pvc_file} -m {segfile_4d} -p RBV+VC -x 4.2 -y 4.2 -z 4.2")
    ki_file = sub_vat.generate_outfile_path(method_short='desc-patlak-slope')
    #ki_pvc_file = sub_vat.generate_outfile_path(method_short='desc-ki_pvc-rbv')
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/pvc_make4d -i {freesurfer_file} -o {segfile_4d}")
    #os.system(f"/home/usr/odonnellj/PETPVC-build/src/petpvc -i {ki_file} -o {ki_pvc_file} -m {segfile_4d} -p RBV+VC -x 4.2 -y 4.2 -z 4.2")


    # reg ki to atlas
    ref = ants.image_read('/data/jsp/human2/AaronProjects/PRISMA_TRIO_PIB_NL/PRISMA_TRIO_PIB_NL_MNI152_T1_0p9mm.nii.gz')
    ki_ants = ants.image_read(ki_file)
    ki_warp = ants.apply_transforms(
        fixed=ref,
        moving=ki_ants,
        transformlist=[atlas_warp_file],
        imagetype=0,
        verbose=1
    )
    ki_smooth = ants.smooth_image(image=ki_warp,sigma=4.2,FWHM=True)
    ants.image_write(ki_smooth,sub_vat.generate_outfile_path(method_short='desc-ki_space-atlas'))

    #suvr_ants = ants.image_read(suvr_pvc_file)
    #suvr_warp = ants.apply_transforms(
    #    fixed=ref,
    #    moving=suvr_ants,
    #    transformlist=[atlas_warp_file],
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
    parser.add_argument('--skip',required=False,help='List of steps to skip',nargs='+',default=[])
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
        vat_protocol(sub,args.out_dir,pet_dir,reg_dir,skip=args.skip)
        #try:
        #    vat_protocol(sub,args.out_dir,pet_dir,reg_dir,skip=True)
        #except:
        #    print(f"Couldn't run {sub}, passing")
        #    pass

main()
