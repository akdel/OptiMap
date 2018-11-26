import numpy as np
import numba as nb
from math import sqrt
from OptiAsm import assembly_utils as au
from scipy import ndimage

def clip_by_zoom(mol, reduce_to=0.66):
    return ndimage.zoom(mol, reduce_to, order=1)

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

def section_vs_section(section1, section2, fft_molecules, fft_rev_molecules, maxes, width=40, top=10):
    number_of_molecules, length = fft_molecules.shape
    results = np.zeros((number_of_molecules, top), dtype=np.float64)
    section2 = np.array(list(section2)).astype(int)
    db_molecules = fft_molecules[section2[0]:section2[1]]
    for i in range(section1[0], section1[1], width):
        print(i)
        subject_molecules = fft_molecules[i:i+width]
        rev_subject_molecules = fft_rev_molecules[i:i+width]
        multiple_top_corrs = get_multiple_products(subject_molecules, rev_subject_molecules, db_molecules)
        mol_range = np.arange(i, i + width)
        normalized_top_corrs = np.zeros(multiple_top_corrs.shape)
        numba_normalize_molecule_correlation_array(multiple_top_corrs, maxes, section2, mol_range, normalized_top_corrs)
        del multiple_top_corrs, mol_range
        numba_arg_sort(normalized_top_corrs, results[i:i+width], top, int(section2[0]))
    return results[section1[0]:section1[1]]


def create_and_link_paired_matrices(period, fft_molecules, fft_rev_molecules, maxes, width=40, top=10, depth=1):
    number_of_molecules, length = fft_molecules.shape
    pair_sets = list()
    for i in range(0, number_of_molecules - (period*2), period):
        current_section1 = (i, i + period)
        current_section2 = (i + period, i + (2 * period))
        argsorted = section_vs_section(current_section1, current_section2, fft_molecules, fft_rev_molecules, maxes, width=width, top=top).astype(int)
        pair_sets.append(from_argsort_to_pairs(argsorted, current_section1, set()))
        del argsorted
    return merge_and_extend_pairs(pair_sets, depth=depth)


# @nb.jit
def get_multiple_products(fft_subject_molecules, fft_subject_rev_molecules, fft_molecules):
    multiple_corr_maxes = np.zeros((fft_subject_molecules.shape[0], fft_molecules.shape[0]), dtype=float)
    # fft_products = np.zeros((2,fft_molecules.shape[0],fft_molecules.shape[1]), dtype=complex)
    fft_products1 = np.zeros((fft_molecules.shape[0],fft_molecules.shape[1]), dtype=complex)
    fft_products2 = np.zeros((fft_molecules.shape[0],fft_molecules.shape[1]), dtype=complex)
    # corr_products = np.zeros((2,fft_molecules.shape[0],fft_molecules.shape[1]), dtype=float)
    corr_maxes = np.zeros(fft_molecules.shape[0])
    for i in range(fft_subject_molecules.shape[0]):
        print("running numba_get_products")
        numba_get_products(fft_subject_molecules[i], fft_subject_rev_molecules[i], fft_molecules, fft_products1, fft_products2)
        # print("running ifft")
        # corr_products[0] = np.fft.ifft(fft_products[0]).real
        # corr_products[1] = np.fft.ifft(fft_products[1]).real
        print("running get_corr_maxes and ifft")
        numba_get_corr_maxes(np.fft.ifft(fft_products1).real, np.fft.ifft(fft_products2).real, corr_maxes)
        multiple_corr_maxes[i,:] = corr_maxes
    return multiple_corr_maxes


@nb.njit(parallel=True)
def numba_get_corr_maxes(corr_products1, corr_products2, corr_maxes):
    for i in nb.prange(corr_maxes.shape[0]):
        max_forward = np.max(corr_products1[i,:])
        max_reverse = np.max(corr_products2[i,:])
        current_max = max(max_forward, max_reverse)
        difference = abs(max_forward - max_reverse)
        corr_maxes[i] = current_max + difference


@nb.njit(parallel=True)
def numba_get_products(fft_subject, fft_subject_rev, fft_molecules, fft_products1, fft_products2):
    for i in nb.prange(fft_molecules.shape[0]):
        numba_product(fft_products1[i], fft_subject, fft_molecules[i])
        numba_product(fft_products2[i], fft_subject_rev, fft_molecules[i])


@nb.njit(fastmath=True)
def numba_product(res, a1, a2):
    for i in range(res.shape[0]):
        res[i] = a1[i] * a2[i]

def numpy_ifft(fft_products):
    return np.fft.ifft(fft_products).astype(np.float64)


@nb.njit(parallel=True)
def numba_arg_sort(correlation_scores, results_array, limit, shift):
    for i in nb.prange(correlation_scores.shape[0]):
        results_array[i] = np.argsort(correlation_scores[i])[::-1][:limit] + shift


@nb.njit(parallel=True)
def numba_normalize_molecule_correlation_array(correlation_array, maxes, max_range, mol_range, normalized_array):
    max_segment = maxes[max_range[0]:max_range[1]]
    for i in nb.prange(correlation_array.shape[0]):
        numba_normalize_single_array(correlation_array[i], normalized_array[i], max_segment, maxes[mol_range[i]])


@nb.njit
def numba_normalize_single_array(single_array, result_array, maxes, current_max):
    for i in range(maxes.shape[0]):
        result_array[i] = single_array[i]/sqrt(current_max * maxes[i])


def from_argsort_to_pairs(argsorted, section1, black_list):
    pairs = set()
    start_query = section1[0]
    for i in range(argsorted.shape[0]):
        if i not in black_list:
            query_id = i + start_query
            for j in range(argsorted.shape[1]):
                current = argsorted[i,j]
                if current not in black_list and current != query_id:
                    pairs.add(tuple(sorted([query_id, argsorted[i,j]])))
                else:
                    continue
        else:
            continue
    print("number of pairs found for section", section1, "is: ", len(pairs))
    return pairs


def merge_and_extend_pairs(pair_sets, depth=1):
    all_pairs = []
    for pair_set in pair_sets:
        all_pairs += list(pair_set)
    print("number of all pairs before merging: ", len(all_pairs))
    set_graph = au.set_graph_from_edges(all_pairs)
    print(len(list(set_graph.keys())))
    extended_graph = au.increase_graph_density_extender(set_graph, depth=depth)
    print(len(list(extended_graph.keys())))
    extended_pairs = au.get_pairs_from_graph(extended_graph)
    print("number of all pairs after merging: ", len(extended_pairs))
    return au.get_pairs_from_graph(extended_graph)


def detect_repeat(nick_coordinates, diff_thr=1., num_thr=5):
    diff = np.abs(nick_coordinates[:-1] - nick_coordinates[1:])
    diff = np.abs(diff[:-1] - diff[1:])
    if diff[diff <= diff_thr].shape[0] >= num_thr:
        return True
    else:
        return False


def run_section_for_all_vs_all(fft_molecules: np.ndarray, fft_rev_molecules: np.ndarray, maxes: np.ndarray, 
                               mol_range: (int, int), output_folder: str, file_prefix: str):
    file_name = file_prefix + ("_%s-%s.npy" % mol_range)
    section = section_vs_section(mol_range, (0, fft_molecules.shape[0]), fft_molecules, fft_rev_molecules, maxes, width=1000, top=300)
    np.save(output_folder + file_name, section)


def run_fft_and_max_script_main(molecules: list):
    maxes = np.array([np.dot(x, x) for x in molecules])
    fft_mols, fft_rev_mols = create_forward_fft(molecules, max_len=512)
    return fft_mols, fft_rev_mols, maxes


if __name__ == "__main__":
    from sys import argv
    molecule_path = argv[1]
    _mol_range = (int(argv[2]), int(argv[3]))
    _out_dir = argv[4]
    _molecules = list(np.load(molecule_path))
    _fft_mols, _fft_rev_mols, _maxes = run_fft_and_max_script_main(_molecules)
    del _molecules
    run_section_for_all_vs_all(_fft_mols, _fft_rev_mols, _maxes, _mol_range, _out_dir, "argsorted")

    """molecules = np.arange(1000*512).reshape((1000,512)).astype(float)
    fft_molecules = np.fft.fft(molecules)
    rev_fft_molecules = np.fft.fft(molecules)
    maxes = np.array([np.sum(x**2) for x in molecules])
    print(
        section_vs_section((0,10), (0, 10), fft_molecules, rev_fft_molecules, maxes, width=10, top=5)
    )"""