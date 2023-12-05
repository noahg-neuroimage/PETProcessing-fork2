import argparse
import pathlib
import image_derived_input_function as idif
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # carotid_mask = idif.extract_from_nii_as_numpy(
    #     file_path="/Users/furqandar/Desktop/Work/BrierLab/PracticeData/1179306_v1/car1_2.nii.gz")
    # print(carotid_mask.shape)
    #
    # pvc_images = idif.extract_from_nii_as_numpy(
    #     file_path="/Users/furqandar/Desktop/Work/BrierLab/PracticeData/1179306_v1/1179306_v1_STC_PVC.nii.gz")
    # print(pvc_images.shape)
    #
    # print(idif.compute_average_over_mask(mask=carotid_mask, image=pvc_images[:, :, :, -1]))
    #
    # avg_vals = idif.compute_average_over_mask_of_multiple_frames(mask=carotid_mask, image_series=pvc_images, start=0,
    #                                                              step=1, stop=40)
    #
    # print(avg_vals)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--mask", )
    parser.add_argument("-i", "--images")

    parser.add_argument("-b", "--start", type=int)
    parser.add_argument("-e", '--stop', type=int)
    parser.add_argument("-s", "--step", type=int)

    parser.add_argument("-o", "--outfile")
    parser.add_argument("-p", "--print", action="store_true")
    
    args = parser.parse_args()
    
    assert pathlib.Path(args.mask).is_file(), "Mask file path is incorrect or does not exist."
    assert pathlib.Path(args.images).is_file(), "Images file path is incorrect or does not exist."

    mask = idif.extract_from_nii_as_numpy(file_path=args.mask)
    print(mask.shape)
    images = idif.extract_from_nii_as_numpy(file_path=args.images)
    print(images.shape)

    avg_vals = idif.compute_average_over_mask_of_multiple_frames(mask=mask, image_series=images,
                                                                 start=args.start, stop=args.stop, step=args.step)

    np.savetxt(fname=args.outfile, X=avg_vals.T)
    
    if args.print:
        print(avg_vals)