# from OptiSpeed.psmatch.encode import Encoder
# from OptiSpeed.compress import Compressor
# from OptiSpeed import test_matching as tm
import numpy as np
from LSH.lsh import VectorsInLSH
import numba as nb
import time
# from igraph import *


class LshForMatch(VectorsInLSH):
    def __init__(self, table_shape, encoded_array, custom_table):
        VectorsInLSH.__init__(self, table_shape, encoded_array, custom_table=custom_table)
        self.mapped_bins = dict()


class Grouper:
    """
    Grouper class uses locality sensitive hashing to bin the encoded signals.
    """
    def __init__(self, encoded_array,molecule_boundaries, custom_table=None, lsh_depth=8):
        if custom_table is not None:
            self.lsh = LshForMatch(custom_table.shape[0], encoded_array, custom_table=custom_table)
            self.molecule_kbits = np.zeros((encoded_array.shape[0],custom_table.shape[0]))
        else:
            self.lsh = LshForMatch(lsh_depth, encoded_array, custom_table)
            self.molecule_kbits = np.zeros((encoded_array.shape[0], lsh_depth))
        self.molecule_lists = list()
        self.molecule_kbits = np.unpackbits(self.lsh.search_results.view("uint8")).reshape((self.molecule_kbits.shape[0], -1))
        self.boundaries = molecule_boundaries
        self.molecule_sets = list()
        self.segment_lens = np.zeros(len(molecule_boundaries))
        self._process_to_molecule_sets()
        for i in range(len(molecule_boundaries)):
            start, end = molecule_boundaries[i]
            for j in range(start, end):
                if self.lsh.search_results[j] in self.lsh.mapped_bins:
                    self.lsh.mapped_bins[self.lsh.search_results[j]].append(i)
                else:
                    self.lsh.mapped_bins[self.lsh.search_results[j]] = [i]

    def _process_to_molecule_sets(self):
        i = 0
        for boundary_start, boundary_end in self.boundaries:
            self.molecule_sets.append(set([r for r in self.lsh.search_results[boundary_start:boundary_end]]))
            self.molecule_lists.append(self.lsh.search_results[boundary_start:boundary_end].copy())
            self.segment_lens[i] = abs(boundary_end - boundary_start)
            i += 1


class PsMatch(Grouper):
    """
    This class will use the LSH group ids per molecule to compute the pairwise jaccard distance between each molecule.
    """
    def __init__(self, encoded_array, molecule_boundaries, custom_table=None, lsh_depth=8):
        Grouper.__init__(self, encoded_array, molecule_boundaries, custom_table=custom_table, lsh_depth=lsh_depth)
        self.pairwise_jaccard = None
        self.log_mols = None
        self.nbits = custom_table.shape[0]

    @classmethod
    def without_autoencoding(cls, molecules, length, log_snr=1.2, snr=2.9, thr=3, verbose=True, nbits=16):
        def verbose_print(text):
            if verbose:
                print(text)
            else:
                pass

        def empty_model(length=100):
            return None

        def create_randoms(nbits=16, l=length):
            randoms = np.zeros((nbits, l))
            steps = l//nbits
            for i in range(1, nbits):
                x = np.zeros(l)
                x[i * steps:i * steps + 6] = l
                randoms[i] = x
            return randoms

        verbose_print("Segments starting to load")
        enc = Encoder(empty_model, "", molecules, length=length, log_transform=True, log_snr=log_snr, snr=snr)
        verbose_print("Segments loading")
        enc.encoded_segments = list()
        enc.segments_without_encoding()
        enc.segment_count = 0
        enc.assigned_segments = list()
        arr = enc.get_full_encoded_segment_array()
        encoded_mols = enc.encoded_segments
        encoded_segment_count = 0
        for i in range(len(encoded_mols)):
            for _ in range(len(encoded_mols[i])):
                encoded_segment_count += 1
        encoded_segments_flat = np.zeros((encoded_segment_count, length))
        verbose_print("Segments loaded")
        n = 0
        for i in range(0, len(encoded_mols)):
            for j in range(len(encoded_mols[i])):
                encoded_segments_flat[n] += encoded_mols[i][j]
                n += 1
        verbose_print("Segments transformed for matching")
        psm = PsMatch(arr.copy(), enc.assigned_segments, custom_table=(create_randoms(nbits=nbits) - thr))
        psm.log_mols = enc.mols
        verbose_print("PsMatch initiated")
        return psm

    def compute_pairwise_jaccard(self, start=0, lim=5):
        res = np.zeros((min(len(self.molecule_sets), lim), len(self.molecule_sets)))
        for i in range(start, min(len(self.molecule_sets), lim)):
            for j in range(len(self.molecule_sets)):
                res[i, j] = jaccard(self.molecule_sets[i], self.molecule_sets[j])
        return res

    def match_mol_to_db_old(self, mol_id, ban_limit=10, distance=1, normalize="self"):
        scores = np.zeros(self.lsh.bin_ids_used.shape[0], dtype="int32")
        banned = self.lsh.bin_ids_used.view(f"uint{self.nbits}")[np.argsort(self.lsh.bin_counts)[-ban_limit:]]
        query = np.unique(self.lsh.search_results[self.boundaries[mol_id][0]:self.boundaries[mol_id][1]].view(f"uint{self.nbits}"))
        query = filter_by_num(query, banned)
        db = self.lsh.bin_ids_used.view(f"uint{self.nbits}")
        get_scores_bound(scores, query, db, banned, thr=distance)
        mol_ids, counts = self._scores_to_top_matches(scores)
        if normalize == "both":
            return normalize_scores(mol_ids, counts, self.segment_lens[mol_id], self.segment_lens), mol_ids
        elif normalize == "query":
            return normalize_scores_query(mol_ids, counts, self.segment_lens[mol_id], self.segment_lens), mol_ids
        elif normalize == "db":
            return normalize_scores_db(mol_ids, counts, self.segment_lens[mol_id], self.segment_lens), mol_ids
        else:
            return counts, mol_ids

    def match_mol_to_db(self, mol_id, banned, scores, ban_limit=10, distance=1, normalize="self"):
        # scores = np.zeros(self.lsh.bin_ids_used.shape[0], dtype="int32")
        # banned = self.lsh.bin_ids_used.view("uint16")[np.argsort(self.lsh.bin_counts)[-ban_limit:]]
        query = np.unique(self.lsh.search_results[self.boundaries[mol_id][0]:self.boundaries[mol_id][1]].view(f"uint{self.nbits}"))
        query = filter_by_num(query, banned).astype(f"uint{self.nbits}")
        db = self.lsh.bin_ids_used.view(f"uint{self.nbits}")
        get_scores_bound(scores, query, db, banned, thr=distance)
        mol_ids, counts = self._scores_to_top_matches(scores)
        if not mol_ids.shape[0]:
            return counts, mol_ids
        if normalize == "both":
            return normalize_scores(mol_ids, counts, self.segment_lens[mol_id], self.segment_lens), mol_ids
        elif normalize == "query":
            return normalize_scores_query(mol_ids, counts, self.segment_lens[mol_id], self.segment_lens), mol_ids
        elif normalize == "db":
            return normalize_scores_db(mol_ids, counts, self.segment_lens[mol_id], self.segment_lens), mol_ids
        else:
            return counts, mol_ids

    def _scores_to_top_matches(self, scores):
        try:
            matched_mols = np.concatenate([self.lsh.mapped_bins[x] for x in self.lsh.bin_ids_used[scores > 0]])
            return np.unique(matched_mols, return_counts=True)
        except ValueError:
            return np.array([0]), np.array([0])

    def match_kmers_to_db(self, mol_id, ban_limit=10, distance=1):
        scores = np.zeros(len(self.boundaries))
        banned = self.lsh.bin_ids_used.view(f"uint{self.nbits}")[np.argsort(self.lsh.bin_counts)[-ban_limit:]]
        query = self.molecule_lists[mol_id].view(f"uint{self.nbits}")
        db = self.lsh.search_results.view(f"uint{self.nbits}")
        get_scores_v2(scores, query, db, np.array(self.boundaries), banned, thr=distance)
        return scores

    def get_raw_optispeed_pairs(self, method="both", lim=300, ban_limit=7, distance=1, test_limit=100):
        scores = np.zeros(self.lsh.bin_ids_used.shape[0], dtype="int32")
        banned = self.lsh.bin_ids_used.view(f"uint{self.nbits}")[np.argsort(self.lsh.bin_counts)[-ban_limit:]]
        res = np.zeros((test_limit, lim))
        t = time.time()
        for p in range(test_limit):
            if p % 1000 == 0:
                print(time.time() - t)
                print(p)
                t = time.time()
            scores_filt, ids = self.match_mol_to_db(p, banned, scores, ban_limit=ban_limit, distance=distance, normalize=method)
            top = ids[np.argsort(scores_filt)[::-1][:lim]].astype(int)
            res[p, :top.shape[0]] = top
        return res

    def compute_pairs_with_optimap(self, log_mols, method="both", lim=300, ban_limit=7, distance=1, overlap_limit=250, test_limit=100, load=""):
        prerest, prelens = tm.mols_to_mollens(log_mols)
        if len(load):
            all_filter_ids = np.load(load)
        else:
            all_filter_ids = self.get_raw_optispeed_pairs(method=method, lim=lim, ban_limit=ban_limit, distance=distance, test_limit=test_limit)
            np.save("search_results.npy", all_filter_ids)
        return tm.get_filtered_alignments_all_mols(log_mols, prerest, prelens, all_filter_ids.astype(int), limit=overlap_limit, test_limit=test_limit)

    def compute_1vs1_dynamic_bit_alignment(self, mol_id1, mol_id2, overlap=15, gap_penalty=0):
        mol_1_segments = self.lsh.search_results[self.boundaries[mol_id1][0]:self.boundaries[mol_id1][1]].view(f"uint{self.nbits}")
        mol_2_segments = self.lsh.search_results[self.boundaries[mol_id2][0]:self.boundaries[mol_id2][1]].view(f"uint{self.nbits}")
        # print(mol_1_segments,mol_2_segments)
        # print(np.intersect1d(mol_1_segments, mol_2_segments))
        distance_matrix = compute_alignment_distance_matrix(mol_1_segments, mol_2_segments)
        dtw_matrix = make_dtw_matrix(distance_matrix, gap_penalty=gap_penalty)
        return np.min(dtw_matrix[overlap:, overlap:])
        # return dtw_matrix

    # def compute_dynamic_bit_alignments_from_pairs(self, pairs, overlap=12, gap_penalty=1):
    #     return compute_bit_alignment_for_pairs(pairs, self.boundaries, , overlap=12, gap_penalty=1)
    #     pass


class DeBruijn:
    def __init__(self, molecules, length, log_snr=1.2, snr=2.9, thr=3, verbose=True, nbits=64):
        psm = PsMatch.without_autoencoding(molecules, length,
                                           log_snr=log_snr, snr=snr,
                                           thr=thr, verbose=verbose, nbits=nbits)
        self.psm = psm
        f = open("./temp_graph.tsv", "w")
        for i in range(len(psm.boundaries)):
            start, end = psm.boundaries[i]
            for j in range(start, end-1, 2):
                part1 = psm.lsh.search_results.view(f"uint{nbits}")[j]
                part2 = psm.lsh.search_results.view(f"uint{nbits}")[j+1]
                f.write(f"{part1}\t{part2}\t{i}\n")
        f.close()
        self.graph = Graph.Read_Ncol("./temp_graph.tsv", directed=True, weights=True)


@nb.njit(parallel=True)
def compute_bit_alignment_for_pairs(pairs, boundaries, segments, overlap=12, gap_penalty=1):
    res = np.zeros(pairs.shape[0])
    for i in nb.prange(res.shape[0]):
        mol_id1 = pairs[i][0]
        mol_id2 = pairs[i][1]
        mol1_start, mol1_end = boundaries[mol_id1][0], boundaries[mol_id1][1]
        mol2_start, mol2_end = boundaries[mol_id2][0], boundaries[mol_id2][1]
        if mol_id1 == mol_id2 or abs(mol1_start - mol1_end) < (overlap+1) or abs(mol2_start - mol2_end) < (overlap+1):
            res[i] = 1000
        else:
            mol1 = segments[mol1_start:mol1_end]
            mol2 = segments[mol2_start:mol2_end]
            distance_matrix = compute_alignment_distance_matrix(mol1, mol2)
            dtw_matrix = make_dtw_matrix(distance_matrix, gap_penalty=gap_penalty)
            # if dtw_matrix.shape[0] < 12 or dtw_matrix.shape[1] < overlap:
            #     current_overlap = min(dtw_matrix.shape[0], dtw_matrix.shape[1]) - 1
            # else:
            #     current_overlap = overlap
            res[i] = np.min(dtw_matrix[overlap:, overlap:])
    return res


@nb.njit
def compute_alignment_distance_matrix_v2(mol1, mol2):
    res = np.zeros((mol1.shape[0], mol2.shape[0]))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            distance = countSetBits(mol1[i] ^ mol2[j])
            if distance < 3:
                res[i, j] = -(3 - distance)
            else:
                res[i, j] = 1
    return res


@nb.njit
def compute_alignment_distance_matrix(mol1, mol2):
    res = np.zeros((mol1.shape[0], mol2.shape[0]))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = countSetBits(mol1[i] ^ mol2[j])
    return res


@nb.njit(parallel=True)
def normalize_scores(ids, scores, self_len, lens):
    res = np.zeros(ids.shape)
    for i in nb.prange(res.shape[0]):
        current_score = scores[i]
        current_id = ids[i]
        res[i] = current_score/((lens[current_id] + self_len)/2)
    return res


@nb.njit(parallel=True)
def normalize_scores_query(ids, scores, self_len, lens):
    res = np.zeros(ids.shape)
    for i in nb.prange(res.shape[0]):
        current_score = scores[i]
        current_id = ids[i]
        res[i] = current_score/min(lens[current_id], self_len)
    return res


@nb.njit(parallel=True)
def normalize_scores_db(ids, scores, self_len, lens):
    res = np.zeros(ids.shape)
    for i in nb.prange(res.shape[0]):
        current_score = scores[i]
        current_id = ids[i]
        res[i] = current_score/max(lens[current_id], self_len)
    return res


@nb.njit(parallel=True)
def get_scores(scores, distances, boundaries, thr=1, nbits=16):
    for i in nb.prange(scores.shape[0]):
        current = distances[boundaries[i, 0]:boundaries[i, 1]]
        current_len = current.shape[0]
        current_score = 0
        for j in range(current.shape[0]):
            if hamming_weight(current[j], nbits=nbits) == 0:
                current_score += 2
            elif hamming_weight(current[j], nbits=nbits) <= thr:
                current_score += 1
            else:
                continue
        scores[i] = current_score/current_len


@nb.njit
def is_element(n, l):
    for i in range(l.shape[0]):
        if n == l[i]:
            return True
        else:
            continue
    return False


@nb.njit
def filter_by_num(arr, banned):
    n = 0
    res = np.zeros(arr.shape, dtype=np.int64)
    for i in range(arr.shape[0]):
        if not is_element(arr[i], banned):
            res[n] = arr[i]
            n += 1
    return res[:n]


@nb.njit(parallel=True)
def get_scores_bound(scores, q_hashes, db_hashes, banned, thr=1, nbits=16):
    q_hashes = filter_by_num(q_hashes, banned).astype(q_hashes.dtype)
    for j in nb.prange(scores.shape[0]):
        current = db_hashes[j]
        if is_element(current, banned):
            scores[j] = 0
        else:
            if hamming_weight(q_hashes ^ current, res=nbits) == 0:
                scores[j] = 1
            elif hamming_weight(q_hashes ^ current, res=nbits) <= thr:
                scores[j] = 1
            else:
                scores[j] = 0


@nb.njit(parallel=True)
def get_scores_v2(scores, q_hashes, db_hashes, boundaries, banned, thr=1, nbits=16):
    # q_hashes = filter_by_num(q_hashes, banned)
    for i in nb.prange(scores.shape[0]):
        current = db_hashes[boundaries[i,0]:boundaries[i,1]]
        current = filter_by_num(current, banned, nbits=nbits)
        current_len = current.shape[0]
        current_score = 0
        for j in range(current.shape[0]):
            if hamming_weight(q_hashes ^ current[j], res=nbits) == 0:
                current_score += 2
            elif hamming_weight(q_hashes ^ current[j], res=nbits) <= thr:
                current_score += 1
            else:
                continue
        scores[i] = (current_score*2)/(current_len+q_hashes.shape[0])


@nb.njit
def _match_kbits_to_db(qkbits, dbkbits):
    preres = np.zeros((dbkbits.shape[0], qkbits.shape[0]), dtype=np.uint64)
    for i in nb.prange(qkbits.shape[0]):
        for j in nb.prange(dbkbits.shape[0]):
            preres[j, i] = qkbits[i] ^ dbkbits[j]
    return preres


def jaccard(s1: set, s2: set):
    return len(s1.intersection(s2))/len(s1.union(s2))


@nb.njit
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count


@nb.njit
def hamming_weight(test, res=16):
    for i in range(test.shape[0]):
        current = countSetBits(test[i])
        if current < res:
            res = current
        else:
            continue
    return res


@nb.njit
def make_dtw_matrix(distance_matrix: np.ndarray,
                    gap_penalty: float = 0.,
                    window: int = np.inf):
    """
    Make matrix using dynamic time warping

    Parameters
    ----------
    distance_matrix
        matrix of distances between corresponding vectors of the two vector sets; shape = (n, m)
    gap_penalty
    window
        constrains warping path according to Sakoe-Chiba band
    Returns
    -------
    accumulated cost matrix; shape = (n, m)
    """
    n, m = distance_matrix.shape
    window = max(window, abs(n - m))
    matrix = np.zeros((n + 1, m + 1))
    matrix[:, 0] = 0
    matrix[0, :] = 0
    matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window)):
            matrix[i, j] = min(matrix[i - 1, j - 1],
                               matrix[i - 1, j] + gap_penalty,
                               matrix[i, j - 1] + gap_penalty) + distance_matrix[i - 1, j - 1]
    return matrix[1:, 1:]



@nb.njit
def find_optimal_warping_path(matrix: np.ndarray) -> np.ndarray:
    """
    Finds optimal warping path from an accumulated cost matrix

    Parameters
    ----------
    matrix
        accumulated cost matrix returned by `make_dtw_matrix`
        shape = (n, m)

    Returns
    -------
    path; shape = (l, 2)

    """
    n, m = matrix.shape[0] - 1, matrix.shape[1] - 1
    path = np.zeros((n + m, 3), dtype=np.int64)
    path[0, 0], path[0, 1] = n, m
    index = 1
    while not (n == 0 and m == 0):
        if n == 0:
            i, j, r = 0, m - 1, matrix[0, m-1]
        elif m == 0:
            i, j, r = n - 1, 0, matrix[n-1, 0]
        else:
            choices = np.array([(n - 1, m - 1), (n - 1, m), (n, m - 1)])
            m_id = np.argmin(np.array([matrix[n - 1, m - 1], matrix[n - 1, m], matrix[n, m - 1]]))
            i, j, r = choices[m_id][0], choices[m_id][1], matrix[choices[m_id][0], choices[m_id][1]]
        path[index, 0], path[index, 1], path[index, 2] = i, j, r
        n, m = i, j
        index += 1
    return path[: index]

@nb.njit
def unik(arr):
    argsorted = np.argsort(arr)
    aux = arr[argsorted]
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    aux_mask = aux[mask]
    ret = np.empty(aux_mask.shape[0]+1, dtype=np.int64)
    ret[: -1] = np.nonzero(mask)[0]
    ret[-1] = mask.size
    ret[:-1] = np.diff(ret)
    return aux_mask, np.diff(ret)


def find_min_distance():
    pass

if __name__ == "__main__":
    path_to_mols = "/mnt/local_scratch/akdel001/specific_signals.npy"
    mollist = np.load(path_to_mols)[:15000]
    snr = 3.2
    enc = Encoder(Compressor, "/mnt/local_scratch/akdel001/OptiSpeed/output/models/model1", mollist)
    # enc.encode_segments()
    # np.save("/mnt/local_scratch/akdel001/OptiSpeed/output/encoded_segments.npy", enc.encoded_segments)
    enc.load_segments("/mnt/local_scratch/akdel001/OptiSpeed/output/encoded_segments.npy")
    arr = enc.get_full_encoded_segment_array()
    print(arr[:10])
    print(enc.assigned_segments[:10])
    grouper = PsMatch(arr, enc.assigned_segments, lsh_depth=16)
    print(grouper.molecule_sets[:10])
    print(np.argsort(grouper.compute_pairwise_jaccard(lim=1)[0])[::-1][:25])
