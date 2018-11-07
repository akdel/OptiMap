from OptiMap import nb, np, correlation_struct, molecule_struct

@nb.njit(parallel=True)
def _normalized_correlation(mol1, mol2, self_corr1, self_corr2, res):
    total = res.shape[0]
    for i in nb.prange(total):
        if i < mol1.shape[0]:
            mol1_segment = mol1[-i:]
            mol2_segment = mol2[:i]
            mol1_self_segment = self_corr1[-i:]
            mol2_self_segment = self_corr2[:i]
            res[i] = np.sum(mol1_segment*mol2_segment)/np.sum(np.sqrt(mol1_self_segment*mol2_self_segment))
        elif i < mol2.shape[0]:
            start = i-mol1.shape[0]
            mol1_segment = mol1
            mol2_segment = mol2[start:start+mol1.shape[0]]
            mol1_self_segment = self_corr1
            mol2_self_segment = self_corr2[start:start+mol1.shape[0]]
            res[i] = np.sum(mol1_segment*mol2_segment)/np.sum(np.sqrt(mol1_self_segment*mol2_self_segment))
        else:
            mol1_segment = mol1[:total-i]
            mol2_segment = mol2[-(total-i):]
            mol1_self_segment = self_corr1[:total-i]
            mol2_self_segment = self_corr2[-(total-i):]
            res[i] = np.sum(mol1_segment*mol2_segment)/np.sum(np.sqrt(mol1_self_segment*mol2_self_segment))

@nb.njit
def normalized_correlation(mol1, mol2):
    self_corr1 = mol1**2
    self_corr2 = mol2**2
    res = np.zeros(mol1.shape[0] + mol2.shape[0])
    _normalized_correlation(mol1, mol2, self_corr1, self_corr2, res)
    return res