from OptiMap import nb, np, correlation_struct, molecule_struct


@nb.njit
def _normalized_correlation(mol1, mol2, mol1_max, mol2_max):
    max_both = np.sqrt(mol1_max * mol2_max)
    corr = np.correlate(mol1, mol2)
    return corr/max_both

@nb.njit
def normalized_correlation(mol1, mol2):
    auto_corr_mol1 = np.correlate(mol1, mol1)
    auto_corr_mol2 = np.correlate(mol2, mol2)
    return _normalized_correlation(mol1, mol2, auto_corr_mol1, auto_corr_mol2)
