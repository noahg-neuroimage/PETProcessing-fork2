import argparse
import numpy as np
from .idif_necktangle import (
    single_threshold_idif_from_4d_pet_with_necktangle,
    double_threshold_idif_from_4d_pet_necktangle,
    get_frame_time_midpoints,
    load_fslmeants_to_numpy_3d
)

def main():
    parser = argparse.ArgumentParser(description="Calculate IDIF using necktangle methods.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Single threshold sub-command
    single_parser = subparsers.add_parser("single", help="Use single threshold method")
    single_parser.add_argument("pet_4d_filepath", type=str, help="Path to the input 4D PET image file (numpy format).")
    single_parser.add_argument("carotid_necktangle_mask_3d_filepath", type=str, help="Path to the carotid mask file (numpy format).")
    single_parser.add_argument("percentile", type=float, help="Percentile for threshold value.")
    single_parser.add_argument("--bolus_start_frame", type=int, default=3, help="Bolus start frame (default: 3).")
    single_parser.add_argument("--bolus_end_frame", type=int, default=7, help="Bolus end frame (default: 7).")

    # Double threshold sub-command
    double_parser = subparsers.add_parser("double", help="Use double threshold method")
    double_parser.add_argument("necktangle_filepath", type=str, help="Path to the 4D PET necktangle matrix file (numpy format).")
    double_parser.add_argument("percentile", type=float, help="Percentile for manual threshold value.")
    double_parser.add_argument("frame_start_times_filepath", type=str, help="Path to the frame start times file (numpy format).")
    double_parser.add_argument("frame_duration_times_filepath", type=str, help="Path to the frame duration times file (numpy format).")

    args = parser.parse_args()

    if args.command == "single":
        pet_4d_data = np.load(args.pet_4d_filepath)
        carotid_necktangle_mask_3d_data = np.load(args.carotid_necktangle_mask_3d_filepath)
        tac = single_threshold_idif_from_4d_pet_with_necktangle(
            pet_4d_data=pet_4d_data,
            carotid_necktangle_mask_3d_data=carotid_necktangle_mask_3d_data,
            percentile=args.percentile,
            bolus_start_frame=args.bolus_start_frame,
            bolus_end_frame=args.bolus_end_frame
        )
        np.savetxt("single_threshold_tac.csv", tac, delimiter=",")
        print("Single threshold IDIF calculation complete. TAC saved to single_threshold_tac.csv")

    elif args.command == "double":
        necktangle_matrix = np.load(args.necktangle_filepath)
        frame_start_times = np.load(args.frame_start_times_filepath)
        frame_duration_times = np.load(args.frame_duration_times_filepath)
        frame_midpoint_times = get_frame_time_midpoints(frame_start_times, frame_duration_times)
        tac = double_threshold_idif_from_4d_pet_necktangle(
            necktangle_matrix=necktangle_matrix,
            percentile=args.percentile,
            frame_midpoint_times=frame_midpoint_times
        )
        np.savetxt("double_threshold_tac.csv", tac, delimiter=",")
        print("Double threshold IDIF calculation complete. TAC saved to double_threshold_tac.csv")

if __name__ == "__main__":
    main()
