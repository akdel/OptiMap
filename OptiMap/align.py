# from OptiMap import nb, np, correlation_struct, molecule_struct
import numpy as np
import numba as nb
from math import sqrt

@nb.njit
def _normalized_correlation(mol1, mol2, self_corr1, self_corr2, res, total, limit):
    for i in range(limit,total-limit):
        if i < mol1.shape[0]:
            mol1_segment = mol1[-i:]
            mol2_segment = mol2[:i]
            mol1_self_segment = np.dot(mol1[-i:],mol1[-i:])
            mol2_self_segment = np.dot(mol2[:i],mol2[:i])
        elif i < mol2.shape[0]:
            start = i-mol1.shape[0]
            mol1_segment = mol1
            mol2_segment = mol2[start:start+mol1.shape[0]]
            mol1_self_segment = np.dot(mol1,mol1)
            mol2_self_segment = np.dot(mol2[start:start+mol1.shape[0]],mol2[start:start+mol1.shape[0]])
        else:
            mol1_segment = mol1[:total-i]
            mol2_segment = mol2[-(total-i):]
            mol1_self_segment = np.dot(mol1[:total-i],mol1[:total-i])
            mol2_self_segment = np.dot(mol2[-(total-i):],mol2[-(total-i):])
        res[i] = np.dot(mol1_segment,mol2_segment)/sqrt(mol1_self_segment*mol2_self_segment)

@nb.njit
def normalized_correlation(mol1, mol2, limit=100):
    self_corr1 = mol1**2
    self_corr2 = mol2**2
    res = np.zeros(mol1.shape[0] + mol2.shape[0], dtype=np.float64)
    _normalized_correlation(mol2, mol1, self_corr1, self_corr2, res, res.shape[0], limit)
    return res
