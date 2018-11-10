import numpy as np
import numba as nb


# The idea is to do the fft and ifft in numpy and the rest in parallel with numba.

# This main function (section_vs_section) should do these:
# 1. 



def prepare_fft_molecules(molecules, fft_compression_ratio=0.3):
    pass

def section_vs_section(section1, section2, fft_molecules, maxes):
    for i in range(section1[0], section1[1]):
        for j in range(section2[0], section2[1]):
            pass
    pass


@nb.njit(parallel=True)
def numba_get_products(fft_subject, fft_molecules, fft_products):
    return fft_products


def numpy_ifft(fft_products):
    return np.fft.ifft(fft_products).astype(np.float64)


@nb.njit(parallel=True)
def numba_get_maxes(correlation_results, molecule_max_array):
    return molecule_max_array

@nb.njit(parallel=True)
def numba_normalize_molecule_correlation_array(correlation_array, maxes, normalized_array):
    return normalized_array