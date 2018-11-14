import numpy as np
import numba as nb
import OptiMap.pairwise as pairwise


def find_maxes(memmap_file, length, width):
    res = np.zeros(length)
    for i in range(0, res.shape[0], width):
        print(start,start+width)
        start = i * (length)
        stop = start + (width * length)
        memmap = get_memmap_width(memmap_file, i*length, (i+1)*length)
        mem = np.array(memmap).reshape((length, -1))
        del memmap
        res[i:i+width] = np.max(mem, axis=0)
        del mem
    return res


def pipe_to_extract(memmap_file, maxes, top, width, number_of_mols):
    sorted_array = np.zeros((number_of_mols, top))
    for i in range(0, number_of_mols, width):
        width_array = np.zeros((width, number_of_mols))
        start = i * (number_of_mols)
        stop = start + (width * number_of_mols)
        memmap = get_memmap_width(memmap_file, start, stop)
        mol_range = np.arange(i, i+width)
        mem = np.array(memmap).reshape((width, -1))
        pairwise.numba_normalize_molecule_correlation_array(mem, maxes, (0, number_of_mols), mol_range, width_array)
        del mem, memmap
        pairwise.numba_arg_sort(width_array, sorted_array[i:i+width], top, 0)
    return sorted_array


def get_memmap_width(filename, start, stop):
    mem = np.memmap(filename, dtype=np.float64, mode="r", offset=start*8, shape=(stop-start,))
    return mem