import image_derived_input_function as idif

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    carotid_mask = idif.extract_from_nii_as_numpy(
        file_path="/Users/furqandar/Desktop/Work/BrierLab/PracticeData/1179306_v1/car1_2.nii.gz")
    print(carotid_mask.shape)
    
    pvc_images = idif.extract_from_nii_as_numpy(
        file_path="/Users/furqandar/Desktop/Work/BrierLab/PracticeData/1179306_v1/1179306_v1_STC_PVC.nii.gz")
    print(pvc_images.shape)
    
    print(idif.compute_average_over_mask(mask=carotid_mask, image=pvc_images[:, :, :, -1]))
    
    avg_vals = idif.compute_average_over_mask_of_multiple_frames(mask=carotid_mask, image_series=pvc_images, start=0,
                                                                 stride=1, end=40)
    
    print(avg_vals)
