import numpy as np
import numba as nb
from math import sqrt

def clip_by_fft(mol, reduce_by=0.3):
    orig = np.fft.fft(mol)
    orig = orig[:int(-orig.shape[0]*reduce_by)]
    return np.fft.ifft(orig).real

def create_forward_fft(mols, max_len=512):
    fft_mols = np.zeros((len(mols), max_len*2))
    fft_mols_rev = np.zeros((len(mols), max_len*2))
    for i in range(len(mols)):
        fft_mols[i,:mols[i].shape[0]] = mols[i]
        fft_mols_rev[i,:mols[i].shape[0]] = mols[i][::-1]
    return np.fft.fft(fft_mols), np.fft.fft(fft_mols_rev)

def section_vs_section(section1, fft_molecules, fft_rev_molecules, maxes, width=40, top=10):
    number_of_molecules, length = fft_molecules.shape
    results = np.zeros((number_of_molecules, top), dtype=np.float64)
    for i in range(section1[0], section1[1], width):
        subject_molecules = fft_molecules[i:i+width]
        rev_subject_molecules = fft_rev_molecules[i:i+width]
        multiple_top_corrs = get_multiple_products(subject_molecules, rev_subject_molecules, fft_molecules)
        mol_range = np.arange(i, i + width)
        normalized_top_corrs = np.zeros(multiple_top_corrs.shape)
        numba_normalize_molecule_correlation_array(multiple_top_corrs, maxes, mol_range, normalized_top_corrs)
        del multiple_top_corrs, mol_range
        numba_arg_sort(normalized_top_corrs, results[i:i+width], top)
    return results

@nb.jit
def get_multiple_products(fft_subject_molecules, fft_subject_rev_molecules, fft_molecules):
    multiple_corr_maxes = np.zeros((fft_subject_molecules.shape[0], fft_molecules.shape[0]), dtype=float)
    fft_products = np.zeros((2,fft_molecules.shape[0],fft_molecules.shape[1]), dtype=complex)
    corr_products = np.zeros((2,fft_molecules.shape[0],fft_molecules.shape[1]), dtype=float)
    corr_maxes = np.zeros(fft_molecules.shape[0])
    for i in range(fft_subject_molecules.shape[0]):
        numba_get_products(fft_subject_molecules[i], fft_subject_rev_molecules[i], fft_molecules, fft_products)
        corr_products[0] = np.fft.ifft(fft_products[0]).real
        corr_products[1] = np.fft.ifft(fft_products[1]).real
        numba_get_corr_maxes(corr_products, corr_maxes)
        multiple_corr_maxes[i,:] = corr_maxes
    return multiple_corr_maxes


@nb.njit(parallel=True)
def numba_get_corr_maxes(corr_products, corr_maxes):
    for i in nb.prange(corr_maxes.shape[0]):
        max_forward = np.max(corr_products[0,i,:])
        max_reverse = np.max(corr_products[1,i,:])
        current_max = max(max_forward, max_reverse)
        # difference = np.abs(max_forward - max_reverse)
        corr_maxes[i] = current_max #+ difference


@nb.njit(parallel=True)
def numba_get_products(fft_subject, fft_subject_rev, fft_molecules, fft_products):
    for i in nb.prange(fft_molecules.shape[0]):
        numba_product(fft_products[0][i], fft_subject, fft_molecules[i])
        numba_product(fft_products[1][i], fft_subject, fft_molecules[i])


@nb.njit(fastmath=True)
def numba_product(res, a1, a2):
    for i in range(res.shape[0]):
        res[i] = a1[i] * a2[i]

def numpy_ifft(fft_products):
    return np.fft.ifft(fft_products).astype(np.float64)


@nb.njit(parallel=True)
def numba_arg_sort(correlation_scores, results_array, limit):
    for i in nb.prange(correlation_scores.shape[0]):
        results_array[i] = np.argsort(correlation_scores[i])[::-1][:limit]


@nb.njit(parallel=True)
def numba_normalize_molecule_correlation_array(correlation_array, maxes, mol_range, normalized_array):
    for i in nb.prange(correlation_array.shape[0]):
        numba_normalize_single_array(correlation_array[i], normalized_array[i], maxes, mol_range[i])


@nb.njit
def numba_normalize_single_array(single_array, result_array, maxes, current_mol):
    current_max = maxes[current_mol]
    for i in range(maxes.shape[0]):
        result_array[i] = single_array[i]/sqrt(current_max * maxes[i])


if __name__ == "__main__":
    molecules = np.arange(1000*512).reshape((1000,512)).astype(float)
    fft_molecules = np.fft.fft(molecules)
    rev_fft_molecules = np.fft.fft(molecules)
    maxes = np.array([np.sum(x**2) for x in molecules])
    print(
        section_vs_section((0,10), fft_molecules, rev_fft_molecules, maxes, width=10, top=5)
    )