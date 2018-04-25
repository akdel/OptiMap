# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libc.math cimport sqrt
cimport cython
from time import time


cdef inline double sum_test(double[:] values) except? -1:
    cdef Py_ssize_t i
    cdef double res = values[0]
    cdef int n = values.shape[0]
    for i in range(1, n):
        res += values[i]
    return res


cdef inline double get_sqrt(double a, double b) nogil except? -1:
    return sqrt(a * b)


cdef inline list create_length_array(int siglenA, int siglenB):
    """
    :param siglenA: length of shorter signal
    :param siglenB: length of longer signal
    :return: list of lengths to be used in normalization.

    int ==> int ==> list

    """
    cdef list begin = range(1, siglenA, 1)
    cdef list mid = [siglenA] * (siglenB - siglenA + 1)
    cdef list end = range(1, siglenA, 1)[::-1]
    return begin + mid + end


cdef inline int[:] create_length_array_v2(int siglenA, int siglenB):
    cdef int begin_len = siglenA - 1
    cdef int total = siglenB + begin_len
    cdef np.ndarray[np.int32_t, ndim=1] lengths = np.empty(total, dtype=np.int32)
    lengths[:begin_len] = np.arange(1,siglenA,1)
    lengths[begin_len:siglenB] = siglenA
    lengths[siglenB:] = np.arange(1,siglenA,1)[::-1]
    return lengths


cdef inline double[:] subtract(double[:] a, double b):
    cdef Py_ssize_t i = 0
    cdef int length = a.shape[0]
    cdef double[:] new_array = np.empty(length, dtype="float64")
    for i in range(length):
        new_array[i] = a[i] - b
    return new_array


cdef inline double[:] square(double[:] a):
    cdef Py_ssize_t i = 0
    cdef int length = a.shape[0]
    cdef double[:] new_array = np.empty(length, dtype="float64")
    for i in range(length):
        new_array[i] = a[i] * a[i]
    return new_array


cdef inline double get_sum_of_squares(double[:] value_list, double m):
    """
    Returns the sum of squares adjusted by the background...
    :param value_list:
    :param m: Signal mean
    :return:

    ndarray ==> np.int ==> np.int

    """
    return sum_test(square(subtract(value_list, m)))


cdef inline double get_corr(double[:] signalA, double[:] signalB, double mA, double mB) except? -1:
    """
    Returns correlation score adjusted by background (using the signal mean)
    :param signalA:
    :param signalB:
    :param mA: mean of signal a
    :param mB: mean of signal b
    :return:

    ndarray ==> ndarray ==> np.int ==> np.int ==> np.int

    """
    return sum_test(np.multiply(subtract(signalA,mA), subtract(signalB,mB)))


cpdef inline double[:] extract_range_into_new_array(double[:] array_in, int start, int stop):
    cdef double[:] _slice = np.empty(stop - start, dtype="float64")
    cdef Py_ssize_t i = 0
    for i in range(start, stop, 1):
        _slice[i - start] = array_in[i]
    return _slice


cdef inline double[:] clip_get_corr_run(double[:] _signal, int length, str ori="left", int secondary=0):
    cdef int siglen = <int>_signal.shape[0]
    if ori == "left":
        return extract_range_into_new_array(_signal, 0, length)
    elif ori == "right":
        return extract_range_into_new_array(_signal, siglen - length, siglen)
    elif ori == "mid":
        return extract_range_into_new_array(_signal, length, secondary)
    else:
        pass


cpdef double[:] normed_xcorr_no_means(double[:] signalA, double[:] signalB, int min_len=100):
    """
    :param signalA: Short signal
    :param signalB: Long signal
    :return:

    ndarray ==> ndarray ==> np.int ==> ndarray

    """
    first = True
    flipped = False
    cdef int top_len = signalB.shape[0]
    cdef int long_len = signalA.shape[0]
    cdef int[:] lengths = create_length_array_v2(top_len, long_len)
    cdef double ssB, ssA, max_idx
    cdef int total_len = top_len + long_len
    cdef double sqrt_div, sqrt_res, corr
    cdef double[:] results = np.zeros(total_len, dtype="float64")
    cdef Py_ssize_t i, r, rr
    cdef int start = min_len
    cdef int end = total_len - min_len
    cdef int _len
    cdef double ssB_complete = get_sum_of_squares_no_means_range(signalB, 0, top_len)
    for i in range(start, end, 1):
        _len = lengths[i]
        if _len != top_len:
            if not flipped:
                ssB = get_sum_of_squares_no_means_range(signalB, top_len-_len, top_len)
                ssA = get_sum_of_squares_no_means_range(signalA, 0, _len)
                corr = get_corr_no_means_range(signalA, 0, _len, signalB, top_len-_len)
            else:
                ssB = get_sum_of_squares_no_means_range(signalB, 0, _len)
                ssA = get_sum_of_squares_no_means_range(signalA, long_len-_len, long_len)
                corr = get_corr_no_means_range(signalA, long_len-_len, long_len, signalB, 0)
            sqrt_res = get_sqrt(ssA, ssB)
            # sqrt_res = ssA + ssB + corr
            sqrt_div = corr / sqrt_res
            results[i] = sqrt_div
        elif _len == top_len:
            flipped = True
            if first:
                r = 1
                first = False
                ssA = get_sum_of_squares_no_means_range(signalA, 0, top_len)
                corr = get_corr_no_means_range(signalA, 0, top_len, signalB, 0)
            else:
                ssA = get_sum_of_squares_no_means_range(signalA, r, r + top_len)
                corr = get_corr_no_means_range(signalA, r,  r + top_len, signalB, 0)
                r += 1
            sqrt_res = get_sqrt(ssA, ssB_complete)
            sqrt_div = corr / sqrt_res
            results[i] = sqrt_div
    return results


cdef inline double[:] multiply(double[:] signalA, double[:] signalB):
    cdef int length = signalA.shape[0]
    cdef Py_ssize_t i =0
    cdef double[:] new_array = np.empty(length, dtype="float64")
    for i in range(length):
        new_array[i] = signalA[i] * signalB[i]
    return new_array


cdef inline double get_corr_no_means(double[:] signalA, double[:] signalB):
    return sum_test(np.multiply(signalA, signalB))


cdef inline double get_corr_no_means_range(double[:] signalA, int a_start, int a_stop, double[:] signalB, int b_start):
    cdef Py_ssize_t i
    cdef int diff = b_start - a_start
    cdef double corr = signalA[a_start] * signalB[b_start]
    for i in range(a_start + 1, a_stop, 1):
        corr += signalA[i] * signalB[i + diff]
    return corr


cdef inline double get_sum_of_squares_no_means(double[:] value_list):
    return sum_test(square(value_list))


cdef inline double get_sum_of_squares_no_means_range(double[:] signal, int start, int stop):
    cdef Py_ssize_t i
    cdef double range_sqr_sum = signal[start]**2
    for i in range(start + 1, stop, 1):
        range_sqr_sum += signal[i]**2
    return range_sqr_sum


cdef class CrossCorrStruct:
    cdef public double highest_score
    cdef public int long_start, long_end, short_start, short_end, contained, score_pass, long_len, short_len


cdef struct CrossStruct:
    double identity_score
    int overlap_score
    int long_start
    int long_end
    int short_start
    int short_end
    int contained
    int score_pass
    int long_len
    int short_len


cpdef CrossStruct xcorr_result_to_struct(double[:] xcorr_result, int short_len, int long_len, int min_len=100, double min_score=75):
    cdef CrossStruct result_struct
    cdef int max_idx = min_len + np.argmax(xcorr_result[min_len:-min_len])
    result_struct.identity_score = max(xcorr_result) * 100
    result_struct.long_len = long_len; result_struct.short_len = short_len
    if (max_idx >= short_len) and (max_idx <= long_len):
        result_struct.short_start = 0
        result_struct.short_end = short_len
        result_struct.overlap_score = -1 * short_len
        result_struct.contained = 1
        result_struct.long_start = max_idx - short_len
        result_struct.long_end = max_idx
    elif max_idx >= long_len:
        result_struct.short_start = 0
        result_struct.short_end = short_len - max_idx + long_len
        result_struct.overlap_score = -1 * (result_struct.short_end)
        result_struct.contained = 0
        result_struct.long_start = max_idx - short_len
        result_struct.long_end = long_len
    elif max_idx < short_len:
        result_struct.short_start = short_len - max_idx
        result_struct.short_end = short_len
        result_struct.contained = 0
        result_struct.long_start = 0
        result_struct.long_end = max_idx
        result_struct.overlap_score = -1 * max_idx
    if result_struct.identity_score >= min_score: result_struct.score_pass = 1
    else: result_struct.score_pass = 0
    return result_struct
