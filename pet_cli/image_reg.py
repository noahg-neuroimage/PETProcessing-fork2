"""
Image registration
"""
import ants
import nibabel
import numpy as np
import h5py

class ImageReg():
    """
    A class, supplies tools to compute and run image registrations.
    Attributes:
    """
    def __init__(
        self,
        verbose: bool=True
    ):
        """
        Args:
            image_path (str): Path to existing Nifti image file.
            verbose (bool): Set to True to print debugging info to shell. Defaults to True.
        """
        self.verbose = verbose


    def reorient_to_ras(
        self,
        image: nibabel.nifti1.Nifti1Image
    ) -> nibabel.nifti1.Nifti1Image:
        """
        Wrapper for the RAS reorientation used to ensure images are oriented the same.

        Args:
            image: Nibabel-type image to write to file.

        Returns:
            The reoriented nifti file.
        """
        reoriented_image = nibabel.as_closest_canonical(image)

        if self.verbose:
            print("(ImageReg): Image has been reoriented to RAS")

        return reoriented_image


    def h5_parse(
        self,
        h5_file: str
    ) -> np.ndarray:
        """
        Parse an h5 transformation file into an affine transform.

        Args:
            h5_path (str): Path to an h5 transformation file.

        Returns:
            xfm_mat (np.ndarray): Affine transform.
        """
        xfm_hdf5 = h5py.File(h5_file)
        xfm_mat  = np.empty((4,4))
        xfm_mat[:,:3]  = xfm_hdf5['TransformGroup']['1']['TransformParameters'][:] \
                   .reshape((4,3))
        xfm_mat[:,3] = [0,0,0,1]
        return xfm_mat


    def rigid_registration(
        self,
        moving_image: nibabel.nifti1.Nifti1Image,
        fixed_image: nibabel.nifti1.Nifti1Image
    ) -> tuple[nibabel.nifti1.Nifti1Image,str,np.ndarray]:
        """
        Register two images under rigid transform assumptions and return the transformed 
        image and parameters.

        Args:
            moving_image (nibabel.nifti1.Nifti1Image): Image to be registered
            fixed_image (nibabel.nifti1.Nifti1Image): Reference image to be registered to

        Returns:
            mov_on_fix (nibabel.nifti1.Nifti1Image): Moving image on reference fixed image
            xfm_file (str): Path to the composite h5 transform written to file. Reference
                            for using h5 files can be found at:
                            https://open.win.ox.ac.uk/pages/fsl/fslpy/fsl.transform.x5.html
            out_mat (np.ndarray): affine transform matrix of parameters from moving to fixed.
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)

        xfm_output = ants.registration(
            moving=moving_image_ants,
            fixed=fixed_image_ants,
            type_of_transform='DenseRigid',
            write_composite_transform=True
        ) # NB: this is a dictionary!


        mov_on_fix_ants = xfm_output['warpedmovout']
        mov_on_fix = ants.to_nibabel(mov_on_fix_ants)
        xfm_file = xfm_output['fwdtransforms']
        out_mat = self.h5_parse(xfm_file)

        return mov_on_fix, xfm_file, out_mat


    def apply_xfm(
        self,
        moving_image: nibabel.nifti1.Nifti1Image,
        fixed_image: nibabel.nifti1.Nifti1Image,
        xfm_path: str
    ) -> nibabel.nifti1.Nifti1Image:
        """
        Register a moving image to a fixed image using a supplied transform.

        Args:
            moving_image (nibabel.nifti1.Nifti1Image): Image to be resampled onto fixed
                reference image.
            fixed_image (nibabel.nifti1.Nifti1Image): Reference image onto which the 
                moving_image is registered.
            xfm_matrix (nibabel.nifti1.Nifti1Image): Ants-style transformation matrix used
                to apply transformation.

        Returns:
            mov_on_fix (nibabel.nifti1.Nifti1Image): Moving image registered onto the fixed image.
        """
        moving_image_ants = ants.from_nibabel(moving_image)
        fixed_image_ants = ants.from_nibabel(fixed_image)
        img_type = len(fixed_image.shape) - 1  # specific to ants: 4D image has type 3

        mov_on_fix_ants = ants.apply_transforms(
            moving=moving_image_ants,
            fixed=fixed_image_ants,
            transformlist=xfm_path,
            imagetype=img_type)

        mov_on_fix = ants.to_nibabel(mov_on_fix_ants)

        return mov_on_fix
