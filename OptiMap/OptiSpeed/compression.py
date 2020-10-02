import numpy as np
from OptiMap import molecule_struct as ms
from OptiMap import correlation_struct as cs
from dataclasses import dataclass
from OptiMap.OptiSpeed import square_wave as sw
import typing as ty
import numba as nb
import itertools
from OptiMap import align
from scipy import ndimage

MoleculeSegmentIndex = ty.Tuple[int, int]
MoleculeIndex = int
SegmentHash = int


@dataclass
class CompressedMolecule:
    square_wave: sw.SquareWave
    segments: ty.Set[SegmentHash] = None

    def get_segment_array(self, nbits: int):
        return np.array(list(self.segments)) \
            .astype(f"uint{nbits}")


@dataclass
class CompressedAndScored:
    molecules: ty.Dict[MoleculeIndex, CompressedMolecule]
    segment_to_molecules: ty.Dict[SegmentHash, ty.List[MoleculeIndex]]
    unique_segments: np.ndarray
    unique_counts: np.ndarray
    nbits: int
    scores_memory: np.ndarray

    @classmethod
    def from_bnx_arrays(cls, bnx_file_name,
                        nbits: int = 32,
                        length: int = 250,
                        width: int = 10,
                        snr: float = 3.5,
                        min_label: int = 5,
                        zoom_factor: int = 500,
                        only_forward: bool = False) -> "CompressedAndScored":
        # forward molecules
        bnx_arrays_forward: ty.Iterator[dict] = (x for x in sw.generate_bnx_arrays(bnx_file_name))
        forward_molecules: ty.List[sw.SquareWave] = [sw.SquareWave.from_bnx_line(x,
                                                                                 reverse=False,
                                                                                 zoom_factor=zoom_factor,
                                                                                 snr=snr,
                                                                                 width=width) for x in
                                                     bnx_arrays_forward if
                                                     len([y for y in x["label_snr"][:-1] if y >= snr]) > min_label]
        # backward molecules
        bnx_arrays_reverse: ty.Iterator[dict] = (x for x in sw.generate_bnx_arrays(bnx_file_name))
        reverse_molecules: ty.List[sw.SquareWave] = [sw.SquareWave.from_bnx_line(x,
                                                                                 reverse=True,
                                                                                 zoom_factor=zoom_factor,
                                                                                 snr=snr,
                                                                                 width=width) for x in
                                                     bnx_arrays_reverse if
                                                     len([y for y in x["label_snr"][:-1] if y >= snr]) > min_label]
        if only_forward:
            return cls.from_waves(forward_molecules,
                                  nbits=nbits,
                                  length=length)
        else:
            return cls.from_waves(forward_molecules + reverse_molecules,
                                  nbits=nbits,
                                  length=length)

    @classmethod
    def from_molecule_structs(cls, optimap_molecules: ty.Iterator[ms.MoleculeNoBack]):

        pass

    @classmethod
    def from_waves(cls, waves: ty.List[sw.SquareWave], nbits: int = 32, length: int = 250, segment_limit: ty.Tuple[float, float] = (0.25, 0.75)) -> "CompressedAndScored":
        all_segments: ty.List[np.ndarray] = list()
        molecules: ty.List[CompressedMolecule] = list()
        segment_to_molecules: ty.Dict[SegmentHash, ty.List[MoleculeIndex]] = dict()
        for wave in waves:
            current_segments = wave.compress(length, nbits, limit=segment_limit)

            if current_segments is None:
                current_segments = list()
            all_segments.append(current_segments)
            molecules.append(CompressedMolecule(wave,
                                                segments={x for x in list(current_segments)}))

            for segment in current_segments:
                if segment not in segment_to_molecules:
                    segment_to_molecules[segment] = [wave.idx]
                else:
                    segment_to_molecules[segment].append(wave.idx)
        unique_segments, segment_counts = np.unique(np.concatenate(all_segments), return_counts=True)
        return cls({x.square_wave.idx: x for x in molecules}, segment_to_molecules, unique_segments,
                   segment_counts, nbits, np.zeros(unique_segments.shape[0]))

    def match_molecule(self,
                       mol_id: int,
                       scores: np.ndarray,
                       thr: int = 1,
                       jaccard: bool = False) -> ty.Dict[MoleculeIndex, int]:
        scores[:] = 0

        get_scores_bound(scores,
                         self.molecules[mol_id].get_segment_array(self.nbits),
                         self.unique_segments
                         .astype(f"uint{self.nbits}"),
                         thr=thr,
                         nbits=self.nbits)
        matched_segments = self.unique_segments[np.where(scores > 0)[0]]
        matched_molecules: ty.Dict[MoleculeIndex, int] = dict()
        for segment_hash in matched_segments:
            matched_molecule_ids: ty.List[MoleculeIndex] = self.segment_to_molecules[segment_hash]
            for molecule_id in matched_molecule_ids:
                if molecule_id not in matched_molecules:
                    matched_molecules[molecule_id] = 1
                else:
                    matched_molecules[molecule_id] += 1
        if jaccard:
            for this_mol_id in matched_molecules:
                current_molecule: CompressedMolecule = self.molecules[mol_id]
                matched_molecules[this_mol_id] /= len(current_molecule.segments | self.molecules[this_mol_id].segments)
        return matched_molecules

    def run_optispeed_and_output_results(self,
                                         filename: str,
                                         hamming_threshold: int = 3,
                                         normalize: bool = True,
                                         match_score_threshold: float = 0.25,
                                         top: int = 10):
        f = open(filename, "w")
        for i, mol in self.molecules.items():
            match_results: ty.Dict[MoleculeIndex, int] = self.match_molecule(i, self.scores_memory,
                                                                             thr=hamming_threshold, jaccard=normalize)
            for other_id, result in sorted(match_results.items(), key=lambda x: x[1], reverse=True)[:top]:

                if result >= match_score_threshold and other_id != i:
                    other_mol = self.molecules[other_id]
                    orientation: bool = (other_mol.square_wave.idx < 0) and (i < 0)
                    orientation = orientation or ((other_mol.square_wave.idx > 0) and (i > 0))
                    f.write(
                        f"{abs(i)}\t{len(mol.segments)}\t{abs(other_id)}\t{len(other_mol.segments)}\t{orientation}\t{round(result, 4)}\n")
        f.close()

    def to_molecule_array(self) -> 'MoleculeArray':
        max_molecule_length: int = max([x.square_wave.wave.shape[0] for x in self.molecules.values()])
        molecule_array: np.ndarray = np.zeros((len(self.molecules), max_molecule_length))
        lengths: np.ndarray = np.zeros(len(self.molecules), dtype=int)
        molid_to_arrayindex: ty.Dict[int, int] = dict()
        arrayindex_to_molid: ty.Dict[int, int] = dict()
        labels: ty.List[ty.List[int]] = list()
        for i, (id_, mol) in enumerate(self.molecules.items()):
            mol: CompressedMolecule
            molecule_array[i][: mol.square_wave.wave.shape[0]] = mol.square_wave.wave
            lengths[i] = mol.square_wave.wave.shape[0]
            molid_to_arrayindex[id_] = i
            arrayindex_to_molid[i] = id_
            labels.append(mol.square_wave.labels)
        return MoleculeArray(molecule_array, molid_to_arrayindex, arrayindex_to_molid, lengths, labels)


@dataclass
class OptiSpeedResult:
    this_mol: int
    that_mol: int
    segments: ty.Tuple[int, int]
    orientation: bool
    score: float

    @classmethod
    def from_optispeed_result_line(cls, line: str):
        data = line.strip().split("\t")
        this_mol, that_mol = int(data[0]), int(data[2])
        segments = (int(data[1]), int(data[3]))
        orientation = True if data[4] == "True" else False
        score = float(data[5])
        return OptiSpeedResult(this_mol, that_mol, segments, orientation, score)

    @property
    def line(self):
        return f"{abs(self.this_mol)}\t{0}\t{abs(self.that_mol)}\t{0}\t{self.orientation}\t{round(self.score, 4)}\n"

@dataclass
class OptiMapResult(OptiSpeedResult):
    long_overlap: ty.Tuple[int, int]
    short_overlap: ty.Tuple[int, int]
    zoom: float

    @property
    def line(self):
        return f"{abs(self.this_mol)}\t{0}\t{abs(self.that_mol)}\t{0}\t{self.orientation}\t{round(self.score, 4)}\t{self.long_overlap[0]}:{self.long_overlap[1]}\t{self.short_overlap[0]}:{self.short_overlap[1]}\t{self.zoom}\n"


Result = ty.Union[OptiSpeedResult, OptiMapResult]
Results = ty.List[Result]


@dataclass
class OptiSpeedResults:
    molecule_matches: ty.Dict[int, Results]

    @classmethod
    def from_optispeed_results_file(cls, filepath: str):
        results = (OptiSpeedResult.from_optispeed_result_line(x) for x in open(filepath, "r"))
        return cls.from_results(results)

    @classmethod
    def empty_results(cls):
        return cls.from_results(list())

    @classmethod
    def from_molecule_array_all_vs_all(cls, molecule_array):
        id_pairs = {tuple(sorted((x, y))) for y in molecule_array.molid_to_arrayindex.keys() for x in molecule_array.molid_to_arrayindex.keys() if abs(x) != abs(y)}
        results = [OptiSpeedResult(this, that, (10, 10), True, 1.) for this, that in id_pairs]
        results += [OptiSpeedResult(this, that, (10, 10), False, 1.) for this, that in id_pairs]
        return cls.from_results(results)

    @classmethod
    def from_results(cls, results: ty.Iterator[Result]):
        molecule_matches: ty.Dict[int, ty.List[Result]] = dict()
        for current_result in results:
            if current_result.this_mol not in molecule_matches:
                molecule_matches[current_result.this_mol] = [current_result]
            else:
                molecule_matches[current_result.this_mol].append(current_result)
            if current_result.that_mol not in molecule_matches:
                molecule_matches[current_result.that_mol] = [current_result]
            else:
                molecule_matches[current_result.that_mol].append(current_result)
        return OptiSpeedResults(molecule_matches)

    def mol_id_to_correlation_results(self, mol_id: int,
                                      molecule_array: 'MoleculeArray',
                                      length_limit: int = 250,
                                      thr: float = 0.8) -> ty.List[OptiSpeedResult]:

        matched_ids: np.ndarray = np.unique(np.array([molecule_array.
                                                     molid_to_arrayindex[x.that_mol] if x.
                                                     orientation else molecule_array.
                                                     molid_to_arrayindex[-x.that_mol] for x in self.
                                                     molecule_matches[mol_id]], dtype=int))
        matched_ids = matched_ids[matched_ids != molecule_array.molid_to_arrayindex[mol_id]]
        current_id: int = molecule_array.molid_to_arrayindex[mol_id]
        signal: np.ndarray = molecule_array.molecule_array[current_id][: molecule_array.lengths[current_id]]
        res = align.get_filtered_alignments(signal,
                                            molecule_array.molecule_array,
                                            molecule_array.lengths,
                                            matched_ids, limit=length_limit)

        map_results: ty.List[OptiSpeedResult] = list()
        for score, matched_id in res:
            if score >= thr:
                orientation = True if molecule_array.arrayindex_to_molid[matched_id] > 0 else False
                #TODO here put the longer molecule first
                map_results.append(OptiSpeedResult(mol_id, molecule_array.arrayindex_to_molid[matched_id],
                                                   (0, 0), orientation, score))
        return map_results

    def to_all_correlation_results(self, molecule_array: 'MoleculeArray',
                                   length_limit: int = 250,
                                   thr: float = 0.8) -> 'OptiSpeedResults':
        map_results: ty.List[OptiSpeedResult] = list()
        for mol_id in self.molecule_matches.keys():
            map_results += self.mol_id_to_correlation_results(mol_id, molecule_array,
                                                              length_limit=length_limit, thr=thr)
        return OptiSpeedResults.from_results(map_results)

    def get_match_pairs(self, orientation: bool = False) -> ty.Set[ty.Tuple[int, int]]:
        pairs: ty.Set[ty.Tuple[int, int]] = set()
        for matches in self.molecule_matches.values():
            for match in matches:
                if orientation:
                    pairs.add((match.this_mol, match.that_mol))
                else:
                    pairs.add((abs(match.this_mol), abs(match.that_mol)))
        return pairs

    def pairs_to_sparse_correlation_results(self,
                                            molecule_array: 'MoleculeArray',
                                            thr: float = 0.8,
                                            pairs: [None, ty.Set[ty.Tuple[int, int]]] = None,
                                            number_of_threads: int = 4,
                                            minimum_overlapping_labels: int = 9) -> 'OptiSpeedResults':
        import ray
        if pairs is None:
            pairs: ty.Set[ty.Tuple[int, int]] = self.get_match_pairs()

        @ray.remote
        def compute_alignment(alignment: cs.CorrelationStruct, molid1: int, molid2: int) -> OptiMapResult:
            alignment.correlate_with_zoom((0.99, 1.01), True)
            if alignment.long_id and not alignment.reversed:
                return OptiMapResult(molid1, molid2,
                                     (0, 0), True, alignment.max_score,
                                     alignment.long_overlap, alignment.short_overlap, alignment.zoom)
            elif alignment.long_id and alignment.reversed:
                return OptiMapResult(molid1, molid2,
                                     (0, 0), False, alignment.max_score,
                                     alignment.long_overlap, alignment.short_overlap, alignment.zoom)
            elif not alignment.long_id and not alignment.reversed:
                return OptiMapResult(molid2, molid1,
                                     (0, 0), True, alignment.max_score,
                                     alignment.long_overlap, alignment.short_overlap, alignment.zoom)
            else:
                return OptiMapResult(molid2, molid1,
                                     (0, 0), False, alignment.max_score,
                                     alignment.long_overlap, alignment.short_overlap, alignment.zoom)

        def make_molecule_struct_from_sq_wave(wave, labels) -> ms.MoleculeNoBack:
            molecule: ms.MoleculeNoBack = ms.MoleculeNoBack(wave, 1.)
            molecule.nick_signal = wave
            molecule.log_signal = wave
            molecule.nick_coordinates = labels
            molecule.nick_snrs = [15.] * len(molecule.nick_coordinates)
            return molecule

        def generate_corr_structs(pairs):
            for pair in pairs:
                mol_pair = list()
                for i in pair:
                    array_id: int = molecule_array.molid_to_arrayindex[i]
                    length: int = molecule_array.lengths[array_id]
                    sq_wave: np.ndarray = molecule_array.molecule_array[array_id][:length]
                    mol_pair.append(make_molecule_struct_from_sq_wave(sq_wave, molecule_array.labels[array_id]))
                yield cs.CorrelationStruct(mol_pair[0], mol_pair[1], minimum_nick_number=minimum_overlapping_labels,
                                           return_new_signals=False), pair

        ray.init(node_ip_address="0.0.0.0", num_cpus=number_of_threads)
        computed_alignments = [result for
                               result
                               in ray.get(
                [compute_alignment.remote(alignment, m1, m2) for (alignment, (m1, m2)) in generate_corr_structs(pairs)])
                               if
                               result.score >= thr]
        ray.shutdown()
        return OptiSpeedResults.from_results(computed_alignments)

    def transitive_closure(self,
                           molecule_array: 'MoleculeArray',
                           length_limit: int = 250,
                           thr: float = 0.8) -> 'OptiSpeedResults':

        original_graph: ty.Dict[int, ty.Set[int]] = {i: {x.that_mol for x in matches} for (i, matches) in
                                                     self.molecule_matches.items()}
        new_pairs: ty.Set[ty.Tuple[int, int]] = set()
        for this, others in original_graph.items():
            for other in others:
                new_pairs |= {tuple(sorted((other, x))) for x in others if x != other}

        def pairs_to_map_results(pairs: ty.Set[ty.Tuple[int, int]]) -> ty.List[OptiSpeedResult]:
            res: ty.List[OptiSpeedResult] = list()
            for x, y in pairs:
                res.append(OptiSpeedResult(x, y, (0, 0), True, 1.0))
                res.append(OptiSpeedResult(x, y, (0, 0), False, 1.0))
            return res

        return OptiSpeedResults. \
            from_results(pairs_to_map_results(new_pairs) + self.get_matches()). \
            to_all_correlation_results(molecule_array, length_limit=length_limit, thr=thr)

    def write_to_file(self, filename: str) -> None:
        f = open(filename, "w")
        first_line: bool = True
        for matches in self.molecule_matches.values():
            for match in matches:
                if first_line and type(match) == OptiMapResult:
                    f.write("#Molecule1ID<int>\t__\tMolecule2ID<int>\t__\tReversed<bool>\tScore<float>\tLongOverlap<Int,Int>\tShortOverlap<Int,Int>\tStretchRatio<float>\n")
                    first_line = False
                elif first_line:
                    f.write("#Molecule1ID<int>\tNumberOfSegmentsMolecule1<Int>\tMolecule2ID<int>\tNumberOfSegmentsMolecule2<Int>\tReversed<bool>\tScore<float>\n")
                    first_line = False
                else:
                    pass
                f.write(match.line)
        f.close()

    def get_matches(self) -> ty.List[OptiSpeedResult]:
        res: ty.List[OptiSpeedResult] = list()
        for v in self.molecule_matches.values():
            res += v
        return res

    def denoise(self, mol_id: int, molecule_array: 'MoleculeArray') -> np.ndarray:
        this_id: int = molecule_array.molid_to_arrayindex[mol_id]
        this_len: int = molecule_array.lengths[this_id]
        this_mol: np.ndarray = np.zeros((1 + len(self.molecule_matches[mol_id]), this_len))
        this_mol[:, :] = np.nan
        this_mol[-1] = molecule_array.molecule_array[this_id, :this_len]
        for i, match in enumerate(self.molecule_matches[mol_id]):
            that_id: int = molecule_array.molid_to_arrayindex[match.that_mol]
            that_len: int = molecule_array.lengths[that_id]
            that_mol: np.ndarray = molecule_array.molecule_array[that_id, :that_len]
            if not match.orientation:
                that_mol = that_mol[::-1]
            if mol_id == match.this_mol and this_len >= (match.zoom * that_len):
                this_mol[i][match.long_overlap[0]: match.long_overlap[1]] = ndimage.zoom(that_mol, match.zoom)[match.short_overlap[0]: match.short_overlap[1]]
        return this_mol

    def denoise_all(self, molecule_array: 'MoleculeArray', min_coverage: int = 3) -> 'MoleculeArray':
        denoised_array: np.ndarray = np.zeros(molecule_array.molecule_array.shape,
                                              dtype=molecule_array.molecule_array.dtype)
        for mol_id in molecule_array.molid_to_arrayindex.keys():
            array_index: int = molecule_array.molid_to_arrayindex[mol_id]
            if mol_id in self.molecule_matches:
                median_array: np.ndarray = self.denoise(mol_id, molecule_array)
                nan_sum: np.ndarray = np.nansum(median_array, axis=1)
                if nan_sum[nan_sum == 0].shape[0] >= min_coverage:
                    denoised_array[array_index, :molecule_array.lengths[array_index]] = np.nanmedian(median_array, axis=0)
                else:
                    denoised_array[array_index] = molecule_array.molecule_array[array_index]
            else:
                denoised_array[array_index] = molecule_array.molecule_array[array_index]
        return MoleculeArray(denoised_array,
                             molecule_array.molid_to_arrayindex,
                             molecule_array.arrayindex_to_molid,
                             molecule_array.lengths,
                             molecule_array.labels)




@dataclass
class MoleculeArray:
    molecule_array: np.ndarray
    molid_to_arrayindex: ty.Dict[int, int]
    arrayindex_to_molid: ty.Dict[int, int]
    lengths: np.ndarray
    labels: ty.List[ty.List[int]]


@nb.njit
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count


@nb.njit
def hamming_weight(test, res=32):
    for i in range(test.shape[0]):
        current = countSetBits(test[i])
        if current < res:
            res = current
        else:
            continue
    return res


@nb.njit(parallel=True)
def get_scores_bound(scores, q_hashes, db_hashes, thr=1, nbits=32):
    for j in nb.prange(scores.shape[0]):
        current = db_hashes[j]
        if hamming_weight(q_hashes ^ current, res=nbits) == 0:
            scores[j] = 1
        elif hamming_weight(q_hashes ^ current, res=nbits) <= thr:
            scores[j] = 1
        else:
            scores[j] = 0


if __name__ == "__main__":
    length_ = 250_000
    labels_ = list(np.sort(np.random.randint(0, length_, 25)).astype(float))
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    from OptiScan import utils

    bnx = utils.BnxParser("data/511699-520213-del.fasta.bnx")
    bnx.read_bnx_file()

    sqs = [sw.SquareWave.from_bnx_line(line, reverse=False, width=8, snr=3.5) for line in bnx.bnx_arrays if
           len(line["labels"][:-1]) > 5]
    # for s in sqs[:10]:
    #     plt.plot(s.wave)
    #     plt.show()
    c = CompressedAndScored.from_waves(sqs, nbits=16, length=length_)
    print(c.match_molecule(0, c.scores_memory, thr=1))
