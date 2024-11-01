"""
This module contains the FitTacWithRTMs class, used to fit kinetic models to a target and
reference Time Activity Curve.
"""
from typing import Union, Callable
from dataclasses import dataclass
import glob
import numpy as np
from petpal.kinetic_modeling.reference_tissue_models import (fit_frtm2_to_tac,
                                                             fit_frtm2_to_tac_with_bounds,
                                                             fit_frtm_to_tac,
                                                             fit_frtm_to_tac_with_bounds,
                                                             fit_mrtm2_2003_to_tac,
                                                             fit_mrtm_2003_to_tac,
                                                             fit_mrtm_original_to_tac,
                                                             fit_srtm2_to_tac,
                                                             fit_srtm2_to_tac_with_bounds,
                                                             fit_srtm_to_tac,
                                                             fit_srtm_to_tac_with_bounds)
from petpal.utils.time_activity_curve import TimeActivityCurveFromFile


def get_rtm_method(method: str, bounds=None):
    r"""Function for obtaining the appropriate reference tissue model.

    This function accepts a string specifying a reference tissue model (RTM) analysis method. It
    returns a reference to the function that performs the selected analysis method.

    - If the method is 'srtm', 'srtm2', 'frtm', or 'frtm2', and bounds are provided, fitting
        functions with bounds are used.
    - If the method is 'srtm', 'srtm2', 'frtm', or 'frtm2', and bounds are not provided, fitting
        functions without bounds are used.
    - If the method is 'mrtm-original', 'mrtm' or 'mrtm2', related fitting methods are utilized.


    Args:
        method (str): The name of the RTM. This should be one of the following strings:
            'srtm', 'srtm2', 'frtm', 'frtm2', 'mrtm-original', 'mrtm' or 'mrtm2'.
        bounds: The bounds on parameters fit during RTM analysis. This value is only used to 
        determine whether to return the method that uses bounds or the unbounded one. Default None.

    Returns:
        function: A reference to the function that performs the corresponding graphical TAC
        analysis. The returned function will take arguments specific to the analysis method, such
        as input TAC values, tissue TAC values, TAC times in minutes, and threshold time in
        minutes.



    Raises:
        ValueError: If the method name is invalid and not one of 'srtm', 'frtm',
            'mrtm-original', 'mrtm' or 'mrtm2'.


    See Also:
        * :func:`fit_srtm_to_tac_with_bounds`
        * :func:`fit_srtm_to_tac`
        * :func:`fit_frtm_to_tac_with_bounds`
        * :func:`fit_frtm_to_tac`
        * :func:`fit_srtm2_to_tac_with_bounds`
        * :func:`fit_srtm2_to_tac`
        * :func:`fit_frtm2_to_tac_with_bounds`
        * :func:`fit_frtm2_to_tac`
        * :func:`fit_mrtm_original_to_tac`
        * :func:`fit_mrtm_2003_to_tac`
        * :func:`fit_mrtm2_2003_to_tac`

    """
    methods_all = ["srtm","srtm2","mrtm-original","mrtm","mrtm2","frtm","frtm2"]
    if method not in methods_all:
        raise ValueError("Invalid method! Must be either 'srtm', 'frtm', 'mrtm-original', "
                        f"'mrtm' or 'mrtm2'. Got {method}.")

    methods_with_bounds = {"srtm": fit_srtm_to_tac_with_bounds,
                           "srtm2": fit_srtm2_to_tac_with_bounds,
                           "frtm": fit_frtm_to_tac_with_bounds,
                           "frtm2": fit_frtm2_to_tac_with_bounds}

    methods_no_bounds = {"srtm": fit_srtm_to_tac,
                         "srtm2": fit_srtm2_to_tac,
                         "mrtm-original": fit_mrtm_original_to_tac,
                         "mrtm": fit_mrtm_2003_to_tac,
                         "mrtm2": fit_mrtm2_2003_to_tac,
                         "frtm": fit_frtm_to_tac,
                         "frtm2": fit_frtm2_to_tac}

    if bounds is not None:
        return methods_with_bounds.get(method)
    return methods_no_bounds.get(method)


def get_rtm_kwargs(method: Callable,
                   bounds: list=None,
                   k2_prime: float=None,
                   t_thresh_in_mins: float=None):
    """
    Function for getting special keyword arguments to be passed on to the provided when used as
    part of an analysis.

    Takes a callable reference tissue model (RTM) method, and optionally a set of bounds, a k2'
    value, and/or a threshold time. The necessary arguments to run the provided method are then
    processed and assigned to their appropriate values in the dictionary ``args_dict``. The 
    function returns ``args_dict`` with any assigned values, which can be passed to ``method`` in
    the form ``**args_dict``.

    Args:
        method (Callable): A method to fit a TAC with an RTM. Expected one of the methods from
            :doc:`petpal.kinetic_modeling.reference_tissue_models`, such as
            :meth:`fit_srtm_to_tac`.
        bounds: The bounds on parameters fit during RTM analysis, if applicable. Expected order is
            as they appear in the original method, e.g. see :meth:`fit_frtm_to_tac_with_bounds`.
            Default None.
        k2_prime: The `k2_prime` value to be used in the analysis, if applicable. Default None.
        t_thresh_in_mins: The threshold time value to be used in the analysis, if applicable.
            Default None.

    Returns:
        args_dict (dict): Dictionary with all keywords necessary to plug into an RTM analysis.
    
    Important:
        If `bounds`,`k2_prime`, `t_thresh_in_mins` are all unset, as they should be for
        :meth:`fit_srtm_to_tac` for example, will return an empty dictionary.
            
    """
    method_args = method.__annotations__.keys()
    args_dict = {}
    if 'k2_prime' in method_args:
        args_dict['k2_prime'] = k2_prime
    if 't_thresh_in_mins' in method_args:
        args_dict['t_thresh_in_mins'] = t_thresh_in_mins
    if 'r1_bounds' in method_args:
        args_dict['r1_bounds'] = bounds[0]
    if 'k2_bounds' in method_args:
        args_dict['k2_bounds'] = bounds[1]
    if 'k2_bounds' in method_args and 'bp_bounds' in method_args:
        args_dict['bp_bounds'] = bounds[2]
    if 'k2_prime' in method_args and 'bp_bounds' in method_args:
        args_dict['bp_bounds'] = bounds[1]
    if 'k2_bounds' in method_args and 'k4_bounds' in method_args:
        args_dict['k3_bounds'] = bounds[2]
        args_dict['k4_bounds'] = bounds[3]
    if 'k2_prime' in method_args and 'k4_bounds' in method_args:
        args_dict['k3_bounds'] = bounds[1]
        args_dict['k4_bounds'] = bounds[2]
    return args_dict


@dataclass
class RTMKwargs:
    """Class for handling RTM inputs and arguments"""
    method: str
    bounds: list
    k2_prime: float
    t_thresh_in_mins: float


class FitTACWithRTMs:
    r"""
    A class used to fit a kinetic model to both a target and a reference Time Activity Curve (TAC).

    The :class:`FitTACWithRTMs` class simplifies the process of kinetic model fitting by providing
    methods for validating input data, choosing a model to fit, and then performing the fit. It
    takes in raw intensity values of TAC for both target and reference regions as inputs, which are
    then used in curve fitting.

    This class supports various kinetic models, including but not limited to: the simplified and
    full reference tissue models (SRTM & FRTM), and the multilinear reference tissue models
    (Orignial MRMT, MRTM & MRTM2). Each model type can be bounded or unbounded.

    The fitting result contains the estimated kinetic parameters depending on the chosen model.

    Attributes:
        tac_times_in_minutes (np.ndarray): The array representing the time-points for both TACs.
        target_tac_vals (np.ndarray): The target TAC values.
        reference_tac_vals (np.ndarray): The reference TAC values.
        method (str): Optional. The kinetic model to use. Defaults to 'mrtm'.
        bounds (np.ndarray): Optional. Parameter bounds for the specified kinetic model. Defaults
            to None.
        t_thresh_in_mins (float): Optional. The times at which the reference TAC was sampled. 
            Defaults to None.
        k2_prime (float): Optional. The estimated efflux rate constant for the non-displaceable 
            compartment. Defaults to None.
        fit_results (np.ndarray): The result of the fit.

    Example:
        The following example shows how to use the :class:`FitTACWithRTMs` class to fit the SRTM to
        a target and reference TAC.

        .. code-block:: python

            import numpy as np
            import petpal.kinetic_modeling.tcms_as_convolutions as pet_tcm
            import petpal.kinetic_modeling.reference_tissue_models as pet_rtms

            # loading the input tac to generate a reference region tac
            input_tac_times, input_tac_vals = np.asarray(np.loadtxt("../../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt").T,
                                                         float)

            # generating a reference region tac
            ref_tac_times, ref_tac_vals = pet_tcm.generate_tac_1tcm_c1_from_tac(tac_times_in_minutes=input_tac_times, tac_vals=input_tac_vals,
                                                                                k1=1.0, k2=0.2)

            # generating an SRTM tac
            srtm_tac_vals = pet_rtms.calc_srtm_tac(tac_times_in_minutes=ref_tac_times, ref_tac_vals=ref_tac_vals, r1=1.0, k2=0.25, bp=3.0)

            rtm_analysis = pet_rtms.FitTACWithRTMs(target_tac_vals=srtm_tac_vals,
                                                tac_times_in_minutes=ref_tac_times,
                                                reference_tac_vals=ref_tac_vals,
                                                method='srtm')

            # performing the fit
            rtm_analysis.fit_tac_to_model()
            fit_results = rtm_analysis.fit_results[1]


    This will give you the kinetic parameter values of the SRTM for the provided TACs.

    See Also:
        * :meth:`validate_bounds`
        * :meth:`validate_method_inputs`
        * :meth:`fit_tac_to_model`

    """
    def __init__(self,
                 tac_times_in_minutes: np.ndarray,
                 target_tac_vals: np.ndarray,
                 reference_tac_vals: np.ndarray,
                 method: str = 'mrtm',
                 bounds: Union[None, np.ndarray] = None,
                 t_thresh_in_mins: float = None,
                 k2_prime: float = None):
        r"""
        Initialize the FitTACWithRTMs object with specified parameters.

        This method sets up input parameters and validates them. We check if the bounds are correct
        for the given 'method', and we make sure that any fitting threshold are defined for the
        MRTM analyses.

        Args:
            tac_times_in_minutes (np.ndarray): The array representing the time-points for both TACs.
            target_tac_vals (np.ndarray): The array representing the target TAC values.
            reference_tac_vals (np.ndarray): The array representing values of the reference TAC.
            method (str, optional): The kinetics method to be used. Default is 'mrtm'.
            bounds (Union[None, np.ndarray], optional): Bounds for kinetic parameters used in
                optimization. None represents absence of bounds. Default is None.
            t_thresh_in_mins (float, optional): Threshold for time separation in minutes. Default
                is None.
            k2_prime (float, optional): The estimated rate constant related to the flush-out rate
                of the reference compartment. Default is None.

        Raises:
            ValueError: If a parameter necessary for chosen method is not provided.
            AssertionError: If rate constant k2_prime is non-positive.
        """

        self.tac_times_in_minutes: np.ndarray = tac_times_in_minutes
        self.target_tac_vals: np.ndarray = target_tac_vals
        self.reference_tac_vals: np.ndarray = reference_tac_vals
        self.method: str = method.lower()
        self.bounds: Union[None, np.ndarray] = bounds
        self.validate_bounds()

        self.t_thresh_in_mins: float = t_thresh_in_mins
        self.k2_prime: float = k2_prime

        self.validate_method_inputs()

        self.fit_results: Union[None, np.ndarray] = None

    def validate_method_inputs(self):
        r"""Validates the inputs for different methods

        This method validates the inputs depending on the chosen method in the object.

        - If the method is of type 'mrtm', it checks if `t_thresh_in_mins` is defined and positive.
        - If the method ends with a '2' (the reduced/modified methods), it checks if `k2_prime` is 
            defined and positive.

        Raises:
            ValueError: If ``t_thresh_in_mins`` is not defined while the method starts with 'mrtm'.
            AssertionError: If ``t_thresh_in_mins`` is not a positive number.
            ValueError: If ``k2_prime`` is not defined while the method ends with '2'.
            AssertionError: If ``k2_prime`` is not a positive number.

        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.method.startswith("mrtm"):
            if self.t_thresh_in_mins is None:
                raise ValueError(
                    "t_t_thresh_in_mins must be defined if method is 'mrtm'")
            else:
                assert self.t_thresh_in_mins >= 0, "t_thresh_in_mins must be a positive number."
        if self.method.endswith("2"):
            if self.k2_prime is None:
                raise ValueError("k2_prime must be defined if we are using the reduced models: "
                                 "FRTM2, SRTM2, and MRTM2.")
            assert self.k2_prime >= 0, "k2_prime must be a positive number."

    def validate_bounds(self):
        r"""Validates the bounds for different methods

        This method validates the shape of the bounds depending on the chosen method in the object.

        - If the method is 'srtm', it checks that bounds shape is (3, 3).
        - If the method is 'frtm', it checks that bounds shape is (4, 3).

        Raises:
            AssertionError: If the bounds shape for method 'srtm' is not (3, 3)
            AssertionError: If the bounds shape for method 'frtm' is not (4, 3).
            ValueError: If the method is not 'srtm' or 'frtm' while providing bounds.

        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.bounds is not None:
            num_params, num_vals = self.bounds.shape
            if self.method == "srtm":
                assert num_params == 3 and num_vals == 3, ("The bounds have the wrong shape. "
                                                           "Bounds must be (start, lo, hi) for each"
                                                           "of the fitting "
                                                           "parameters: r1, k2, bp")
            elif self.method == "frtm":
                assert num_params == 4 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k2, k3, k4")

            elif self.method == "srtm2":
                assert num_params == 2 and num_vals == 3, ("The bounds have the wrong shape. Bounds"
                                                           "must be (start, lo, hi) "
                                                           "for each of the"
                                                           " fitting parameters: r1, bp")
            elif self.method == "frtm2":
                assert num_params == 3 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k3, k4")
            else:
                raise ValueError(f"Invalid method! Must be either 'srtm', 'frtm', 'srtm2' or "
                                 "'frtm2' if bounds are "
                                 f"provided. Got {self.method}.")

    def fit_tac_to_model(self):
        r"""Fits TAC vals to model

        This method fits the target TAC values to the model depending on the chosen method in the
        object.

        - If the method is 'srtm' or 'frtm', and bounds are provided, fitting functions with bounds
            are used.
        - If the method is 'srtm' or 'frtm', and bounds are not provided, fitting functions without
            bounds are used.
        - If the method is 'mrtm-original', 'mrtm' or 'mrtm2', related fitting methods are utilized.

        Raises:
            ValueError: If the method name is invalid and not one of 'srtm', 'frtm',
                'mrtm-original', 'mrtm' or 'mrtm2'.


        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_srtm2_to_tac_with_bounds`
            * :func:`fit_srtm2_to_tac`
            * :func:`fit_frtm2_to_tac_with_bounds`
            * :func:`fit_frtm2_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        rtm_method = get_rtm_method(method=self.method,bounds=self.bounds)
        rtm_kwargs = get_rtm_kwargs(method=rtm_method,
                                    bounds=self.bounds,
                                    k2_prime=self.k2_prime,
                                    t_thresh_in_mins=self.t_thresh_in_mins)
        self.fit_results = rtm_method(tgt_tac_vals=self.target_tac_vals,
                                      ref_tac_times=self.tac_times_in_minutes,
                                      ref_tac_vals=self.reference_tac_vals,
                                      **rtm_kwargs)

class RTMRegionalAnalysis:
    r"""
    A class used to fit a kinetic model to ROI and reference Time Activity Curves (TACs).

    The :class:`RTMRegionalAnalysis` class simplifies the process of kinetic model fitting by 
    providing methods for validating input data, choosing a model to fit, and then performing the
    fits across all regions. It takes in the reference TAC path, the ROI TAC directory path, and
    any necessary method arguments as inputs, which are then used in curve fitting.

    This class supports various kinetic models, including but not limited to: the simplified and
    full reference tissue models (SRTM & FRTM), and the multilinear reference tissue models
    (Orignial MRMT, MRTM & MRTM2). Each model type can be bounded or unbounded.

    The fitting result contains the estimated kinetic parameters depending on the chosen model.

    Attributes:
        tac_times_in_minutes (np.ndarray): The array representing the time-points for both TACs.
        target_tac_vals (np.ndarray): The target TAC values.
        reference_tac_vals (np.ndarray): The reference TAC values.
        method (str): Optional. The kinetic model to use. Defaults to 'mrtm'.
        bounds (np.ndarray): Optional. Parameter bounds for the specified kinetic model. Defaults
            to None.
        t_thresh_in_mins (float): Optional. The times at which the reference TAC was sampled. 
            Defaults to None.
        k2_prime (float): Optional. The estimated efflux rate constant for the non-displaceable 
            compartment. Defaults to None.
        fit_results (np.ndarray): The result of the fit.

    Example:
        The following example shows how to use the :class:`RTMRegionalAnalysis` class to fit the
        MRTM model to a target and reference TAC.

        .. code-block:: python

            ref_tac_example = "../../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt"
            tacs_from_preproc = "/path/to/data/derivatives/PETPAL/sub-001/tacs/"
            rtm_kwargs = RTMKwargs(method='mrtm', t_thresh_in_mins=30)
            regional_mrtm = RTMRegionalAnalysis(reference_tac_path=ref_tac_example,
                                                tacs_dir=tacs_from_preproc,
                                                rtm_inputs=rtm_kwargs)
            parameters_mrtm = regional_mrtm.fit_results

    This will give you the kinetic parameter values of the MRTM for the provided TACs.

    See Also:
        * :meth:`validate_bounds`
        * :meth:`validate_method_inputs`
        * :meth:`fit_tac_to_model`

    """
    def __init__(self,
                 reference_tac_path: str,
                 tacs_dir: str,
                 rtm_inputs: RTMKwargs):
        r"""
        Initialize the FitTACWithRTMs object with specified parameters.

        This method sets up input parameters and validates them. We check if the bounds are correct
        for the given 'method', and we make sure that any fitting threshold are defined for the
        MRTM analyses. Then, ROI tacs are selected and fitted with MRTM.

        Args:
            reference_tac_path (str): Path to reference region TAC file. Must be readable with
                :class:`TimeActivityCurveFromFile`.
            tacs_dir (str): Path to directory containing ROI TACs. Each TAC must have extension
                *.tsv and be readable with :class:`TimeActivityCurveFromFile`.
            rtm_inputs (RTMKwargs): An instance of dataclass :class:`RTMKwargs` containing the
                values for `method`, `bounds`, `k2_prime`, and/or `t_thresh_in_mins` as necessary
                for analysis. At least `method` must be specified.

        Raises:
            ValueError: If a parameter necessary for chosen method is not provided.
            AssertionError: If rate constant k2_prime is non-positive.

        See also:
            * :module:`petpal.kinetic_modeling.reference_tissue_models`
            * :class:`TimeActivityCurveFromFile`

        """
        self.reference_tac = TimeActivityCurveFromFile(tac_path=reference_tac_path)
        self.tacs_dir = tacs_dir
        self.rtm_inputs = rtm_inputs
        self.rtm_method = get_rtm_method(method=self.rtm_inputs.method,
                                         bounds=self.rtm_inputs.bounds)

        self.validate_tacs_dir()
        self.validate_bounds()
        self.validate_method_inputs()
        self.kwargs_dict = self.get_rtm_kwargs_dict_from_class()
        self.fit_results = self.fit_many_tacs_to_model()


    def validate_tacs_dir(self):
        """
        Verifies the provided `tacs_dir` contains TAC files.

        Raises:
            FileNotFoundError: If there are no TAC files found in the directory.
        """
        tacs_file_list = self.get_tacs_file_list()
        if len(tacs_file_list)<1:
            raise FileNotFoundError("tacs_dir must contain at least one TAC file. Check the"
                                    f"contents of the directory: {self.tacs_dir}.")


    def validate_method_inputs(self):
        r"""Validates the inputs for different methods

        This method validates the inputs depending on the chosen method in the object.

        - If the method is of type 'mrtm', it checks if `t_thresh_in_mins` is defined and positive.
        - If the method ends with a '2' (the reduced/modified methods), it checks if `k2_prime` is 
            defined and positive.

        Raises:
            ValueError: If ``t_thresh_in_mins`` is not defined while the method starts with 'mrtm'.
            AssertionError: If ``t_thresh_in_mins`` is not a positive number.
            ValueError: If ``k2_prime`` is not defined while the method ends with '2'.
            AssertionError: If ``k2_prime`` is not a positive number.

        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.rtm_inputs.method.startswith("mrtm"):
            if self.rtm_inputs.t_thresh_in_mins is None:
                raise ValueError(
                    "t_t_thresh_in_mins must be defined if method is 'mrtm'")
            assert self.rtm_inputs.t_thresh_in_mins >= 0, "t_thresh_in_mins must be a positive number."
        if self.rtm_inputs.method.endswith("2"):
            if self.rtm_inputs.k2_prime is None:
                raise ValueError("k2_prime must be defined if we are using the reduced models: "
                                 "FRTM2, SRTM2, and MRTM2.")
            assert self.rtm_inputs.k2_prime >= 0, "k2_prime must be a positive number."

    def validate_bounds(self):
        r"""Validates the bounds for different methods

        This method validates the shape of the bounds depending on the chosen method in the object.

        - If the method is 'srtm', it checks that bounds shape is (3, 3).
        - If the method is 'frtm', it checks that bounds shape is (4, 3).

        Raises:
            AssertionError: If the bounds shape for method 'srtm' is not (3, 3)
            AssertionError: If the bounds shape for method 'frtm' is not (4, 3).
            ValueError: If the method is not 'srtm' or 'frtm' while providing bounds.

        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        if self.rtm_inputs.bounds is not None:
            num_params, num_vals = self.rtm_inputs.bounds.shape
            if self.rtm_inputs.method == "srtm":
                assert num_params == 3 and num_vals == 3, ("The bounds have the wrong shape. "
                                                           "Bounds must be (start, lo, hi) for each"
                                                           "of the fitting "
                                                           "parameters: r1, k2, bp")
            elif self.rtm_inputs.method == "frtm":
                assert num_params == 4 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k2, k3, k4")

            elif self.rtm_inputs.method == "srtm2":
                assert num_params == 2 and num_vals == 3, ("The bounds have the wrong shape. Bounds"
                                                           "must be (start, lo, hi) "
                                                           "for each of the"
                                                           " fitting parameters: r1, bp")
            elif self.rtm_inputs.method == "frtm2":
                assert num_params == 3 and num_vals == 3, (
                    "The bounds have the wrong shape. Bounds must be (start, lo, hi) "
                    "for each of the fitting parameters: r1, k3, k4")
            else:
                raise ValueError(f"Invalid method! Must be either 'srtm', 'frtm', 'srtm2' or "
                                 "'frtm2' if bounds are "
                                 f"provided. Got {self.rtm_inputs.method}.")


    def get_tacs_file_list(self):
        """
        Get a list of TAC paths from a directory. Assume .tsv files.
        """
        tacs_file_list = glob.glob(f"{self.tacs_dir}/*.tsv")
        return tacs_file_list


    def get_tacs_object_list(self, tacs_file_list: list):
        """
        Get a list of TimeActivityCurveFromFile objects from a list of TAC paths.
        """
        tacs_list = [TimeActivityCurveFromFile(tac_path=tac_file) for tac_file in tacs_file_list]
        return tacs_list


    def get_tacs_vals_list_from_object(self, tacs_object_list: list):
        """
        Get a list of numpy arrays containing TAC values from a list of TimeActivityCurveFromFile
        objects.
        """
        tac_vals_list = []
        for tac_object in tacs_object_list:
            tac_vals_list += [tac_object.tac_vals]
        return tac_vals_list


    def get_rtm_kwargs_dict_from_class(self):
        """
        Get a kwargs dict that functions as a direct input into the RTM methods from the values
        stored in ``self.rtm_kwargs``.
        """
        rtm_kwargs_dict = get_rtm_kwargs(method=self.rtm_method,
                                         bounds=self.rtm_inputs.bounds,
                                         k2_prime=self.rtm_inputs.k2_prime,
                                         t_thresh_in_mins=self.rtm_inputs.t_thresh_in_mins)
        return rtm_kwargs_dict


    def fit_tac_to_model(self, target_tac_vals):
        r"""Fits TAC vals to model

        This method fits the target TAC values to the model depending on the chosen method in the
        object.

        - If the method is 'srtm' or 'frtm', and bounds are provided, fitting functions with bounds
            are used.
        - If the method is 'srtm' or 'frtm', and bounds are not provided, fitting functions without
            bounds are used.
        - If the method is 'mrtm-original', 'mrtm' or 'mrtm2', related fitting methods are utilized.

        Raises:
            ValueError: If the method name is invalid and not one of 'srtm', 'frtm',
                'mrtm-original', 'mrtm' or 'mrtm2'.


        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_srtm2_to_tac_with_bounds`
            * :func:`fit_srtm2_to_tac`
            * :func:`fit_frtm2_to_tac_with_bounds`
            * :func:`fit_frtm2_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        ref_tac = self.reference_tac
        self.fit_results = self.rtm_inputs.method(tgt_tac_vals=target_tac_vals,
                                                  ref_tac_times=ref_tac.tac_times_in_minutes,
                                                  ref_tac_vals=ref_tac.tac_vals,
                                                  **self.kwargs_dict)


    def fit_many_tacs_to_model(self):
        r"""Fits many target TACs to a model

        This method fits the target TAC values to the model depending on the chosen method in the
        object.

        - If the method is 'srtm' or 'frtm', and bounds are provided, fitting functions with bounds
            are used.
        - If the method is 'srtm' or 'frtm', and bounds are not provided, fitting functions without
            bounds are used.
        - If the method is 'mrtm-original', 'mrtm' or 'mrtm2', related fitting methods are utilized.

        Raises:
            ValueError: If the method name is invalid and not one of 'srtm', 'frtm',
                'mrtm-original', 'mrtm' or 'mrtm2'.


        See Also:
            * :func:`fit_srtm_to_tac_with_bounds`
            * :func:`fit_srtm_to_tac`
            * :func:`fit_frtm_to_tac_with_bounds`
            * :func:`fit_frtm_to_tac`
            * :func:`fit_srtm2_to_tac_with_bounds`
            * :func:`fit_srtm2_to_tac`
            * :func:`fit_frtm2_to_tac_with_bounds`
            * :func:`fit_frtm2_to_tac`
            * :func:`fit_mrtm_original_to_tac`
            * :func:`fit_mrtm_2003_to_tac`
            * :func:`fit_mrtm2_2003_to_tac`

        """
        ref_tac = self.reference_tac
        tacs_file_list = self.get_tacs_file_list()
        tacs_object_list = self.get_tacs_object_list(tacs_file_list=tacs_file_list)
        target_tac_vals_list = self.get_tacs_vals_list_from_object(tacs_object_list=tacs_object_list)
        fit_results = []
        for target_tac_vals in target_tac_vals_list:
            try:
                fit_results += [self.rtm_method(tac_times_in_minutes=ref_tac.tac_times_in_minutes,
                                                tgt_tac_vals=target_tac_vals,
                                                ref_tac_vals=ref_tac.tac_vals,
                                                **self.kwargs_dict)]
            except ValueError:
                fit_results += [np.nan]
        return fit_results
