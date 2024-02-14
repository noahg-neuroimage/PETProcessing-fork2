"""Module for interpolating Time Activity Curves (TACs). Mostly used for evenly interpolating PET TACs which tend to be
sampled unevenly with respect to time.

"""

import numpy as np
from scipy.interpolate import interp1d as sp_interpolation
from typing import Tuple