try:
    from OptiScan import scan, database, utils
except ImportError:
    raise Warning("Some OptiMap functionalities require OptiScan.")
import numpy as np
import numba as nb

EPSILON = np.finfo(float).eps

