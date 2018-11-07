from OptiMap import *
from OptiMap import molecule_struct as ms
from OptiMap.align import normalized_correlation as cr
import multiprocessing as mp
from scipy import ndimage

EPSILON = 0.0000001

class CorrelationStruct:
    def __init__(self, molecule_1: ms.Molecule, molecule_2: ms.Molecule, minimum_nick_number=9, bylength=False):
        self.long_id = True
        if molecule_1.nick_signal.shape[0] >= molecule_2.nick_signal.shape[0]:
            self.long_molecule = molecule_1
            self.short_molecule = molecule_2
        else:
            self.long_molecule = molecule_2
            self.long_id = False
            self.short_molecule = molecule_1
        self.correlation = None
        self.zoom = 1
        self.short_change = 0
        self.status = False
        self.reversed = False
        self.max_score = 0
        self.max_index = 0
        if bylength:
            self.minimum_overlap_length = bylength
        elif len(molecule_1.nick_coordinates) < minimum_nick_number or len(molecule_2.nick_coordinates) < minimum_nick_number:
            self.minimum_overlap_length = self.short_molecule.nick_signal.shape[0]
        else:
            try:
                self.minimum_overlap_length = self.set_minimum_overlaps(minimum_nick_number)
            except:
                self.minimum_overlap_length = self.short_molecule.nick_signal.shape[0]
        if self.minimum_overlap_length >= self.short_molecule.nick_signal.shape[0]:
            self.fit_for_cross = False
        else:
            self.fit_for_cross = True
        self.short_overlap = [None, None]
        self.long_overlap = [None, None]
        self.new_short = None
        self.new_long = None
        self.secondary_score = None


    def re_correlate_with_original(self):
        if self.status:
            #### Here make a numba function to correlate only the overlapping part found by previous correlation
            ### for this, find out the short and apply the zoom. take the overlap dot products and normalize
            ### record the outcome to secondary score
            pass
        else:
            return None

    def correlate_with_zoom(self, zoom_range: (float, float), log: bool):
        assert not self.status
        if not self.fit_for_cross:
            self.max_score = 0
            return None
        self.status = True
        scores = list()
        if log:
            long_signal = self.long_molecule.log_signal 
            short_signal = self.short_molecule.log_signal
        else:
            long_signal = self.long_molecule.nick_signal + EPSILON
            short_signal = self.short_molecule.nick_signal + EPSILON

        for zoomed_signal, zoomed_signal_flipped, zoom in generate_stretched_signals(zoom_range,
                                                                                     short_signal,
                                                                                     step=0.001):
            if zoomed_signal.shape[0] > long_signal.shape[0]:
                short_change = True
                _long = zoomed_signal
                short = long_signal
            else:
                short_change = False
                short = zoomed_signal
                _long = long_signal

            full_corr = cr(_long, short, limit=self.minimum_overlap_length)
            full_corr_flipped = cr(_long, short[::-1], limit=self.minimum_overlap_length)

            scores.append((np.max(full_corr), np.argmax(full_corr), zoom, full_corr, False, short_change,
                           short, _long))
            scores.append((np.max(full_corr_flipped), np.argmax(full_corr_flipped), zoom, full_corr_flipped, True,
                           short_change, short[::-1], _long))

        self.max_score, self.max_index, self.zoom, self.correlation, self.reversed, self.short_change, self.new_short, self.new_long = max(scores, key=max_function)

        if self.max_index > self.new_short.shape[0]:
            self.long_overlap[0] = self.max_index - self.new_short.shape[0]
            self.long_overlap[1] = self.long_overlap[0] + self.new_short.shape[0]
            if self.long_overlap[1] > self.new_long.shape[0]:
                self.long_overlap[1] = self.new_long.shape[0]
            self.short_overlap[0] = 0
            if self.new_long.shape[0] < self.max_index:
                self.short_overlap[1] = (self.new_long.shape[0] +
                                         self.new_short.shape[0]) - self.max_index
            else:
                self.short_overlap[1] = self.new_short.shape[0]
        else:
            self.short_overlap[0] = self.new_short.shape[0] - self.max_index
            self.short_overlap[1] = self.new_short.shape[0]
            self.long_overlap[0] = 0
            self.long_overlap[1] = self.max_index

        if self.short_change:
            if self.long_id:
                self.long_id = False
            else:
                self.long_id = True
        self.re_correlate_with_original()

    def set_minimum_overlaps(self, minimum_nick_number: int):
        return np.max([self.short_molecule.nick_coordinates[minimum_nick_number],
                       self.short_molecule.nick_signal.shape[0] -
                       self.short_molecule.nick_coordinates[-minimum_nick_number],

                       self.long_molecule.nick_coordinates[minimum_nick_number],
                       self.long_molecule.nick_signal.shape[0] -
                       self.long_molecule.nick_coordinates[-minimum_nick_number]])


def create_pairwise_structs_in_db(database_name: str, snr=6):
    pair_ids = [-1, -1]
    for molecule_1 in ms.create_molecule_stream_from_db(database_name, snr):
        pair_ids[0] += 1
        # pair_ids[1] = 0
        for molecule_2 in ms.create_molecule_stream_from_db(database_name, snr):
            pair_ids[1] += 1
            if pair_ids[0] != pair_ids[1] and pair_ids[0] < pair_ids[1]:
                corr = CorrelationStruct(molecule_1, molecule_2)
                if corr.fit_for_cross:
                    corr.correlate_with_zoom((0.98, 1.02), log=False)
                    yield corr, tuple(pair_ids)


def create_pairwise_structs_from_npfile(filename: str, snr=6.):
    pair_ids = [-1, -1]
    for molecule_1 in ms.yield_molecules_from_saved_np(filename, snr):
        pair_ids[0] += 1
        # pair_ids[1] = 0
        for molecule_2 in ms.yield_molecules_from_saved_np(filename, snr):
            pair_ids[1] += 1
            if pair_ids[0] != pair_ids[1] and pair_ids[0] < pair_ids[1]:
                corr = CorrelationStruct(molecule_1, molecule_2)
                if corr.fit_for_cross:
                    corr.correlate_with_zoom((0.98, 1.02), log=False)
                    yield corr, tuple(pair_ids)


def create_pairwise_structs_from_arrays(signal_list, snr=6.):
    pair_ids = [-1, -1]
    for molecule_1 in ms.yield_molecules_from_signal_list(signal_list, snr):
        pair_ids[0] += 1
        # pair_ids[1] = 0
        for molecule_2 in ms.yield_molecules_from_signal_list(signal_list, snr):
            pair_ids[1] += 1
            if pair_ids[0] != pair_ids[1] and pair_ids[0] < pair_ids[1]:
                corr = CorrelationStruct(molecule_1, molecule_2)
                if corr.fit_for_cross:
                    corr.correlate_with_zoom((0.98, 1.02), log=False)
                    yield corr, tuple(pair_ids)


def create_one_vs_all_structs(single_signal: ms.MoleculeNoBack, molecule_stream: [ms.MoleculeNoBack]):
    pair_ids = [0, -1]
    for molecule in molecule_stream:
        pair_ids[1] += 1
        corr = CorrelationStruct(single_signal, molecule)
        if corr.fit_for_cross:
            corr.correlate_with_zoom((0.98, 1.02), log=False)
            yield corr, tuple(pair_ids)


def save_corrs_to_disk_from_multivsall(multi_list: list, db_location: str, score_threshold: float, filename: str,
                                       snr: float=3.5):
    corrs = []
    if db_location.endswith(".db"):
        streams = [ms.create_molecule_stream_from_db(db_location, snr=snr) for _ in multi_list]
    elif db_location.endswith(".npy"):
        stream = np.load(db_location)
    else:
        raise AssertionError
    for i in range(len(multi_list)):
        mol, _id = multi_list[i]
        if db_location.endswith(".db"):
            for corr, pair_ids in create_one_vs_all_structs(mol, streams[i]):
                real_pair = list(pair_ids)
                real_pair[0] = _id
                if corr.max_score >= score_threshold and _id < real_pair[1]:
                    print(real_pair, corr.max_score)
                    corrs.append((corr, real_pair))
        elif db_location.endswith(".npy"):
            for corr, pair_ids in create_one_vs_all_structs(mol, stream):
                real_pair = list(pair_ids)
                real_pair[0] = _id
                if corr.max_score >= score_threshold and _id < real_pair[1]:
                    print(real_pair, corr.max_score)
                    corrs.append((corr, real_pair))
    np.save("%s.npy" % filename, corrs)


def split_mols_into_threads(db_location: str, number_of_threads: int, snr=3.5):
    n = 0
    if db_location.endswith(".db"):
        for _ in ms.create_molecule_stream_from_db(db_location, snr=snr):
            n += 1
    elif db_location.endswith(".npy"):
        np_data = np.load(db_location)
        for _ in np_data:
            n += 1
    else:
        raise AssertionError
    number_of_mols_per_thread = int(np.floor(n/number_of_threads))
    molid_groups = list()
    for i in range(0, n, number_of_mols_per_thread):
        molid_groups.append(range(i, i + number_of_mols_per_thread, 1))
    return molid_groups


def obtain_mol_group_from_molid_group(db_location: str, molid_group, snr=3.5):
    n = -1
    if db_location.endswith(".db"):
        for mol in ms.create_molecule_stream_from_db(db_location, snr=snr):
            n += 1
            if n in molid_group:
                yield mol, n
    elif db_location.endswith(".npy"):
        np_data = np.load(db_location)
        for mol in np_data:
            n += 1
            if n in molid_group:
                yield mol, n
    else:
        raise AssertionError


def create_range(_from: float, _to: float, step: float):
    n = _from - step
    while n <= _to:
        n += step
        yield n


def generate_matching_pairs(database_name: str, score_threshold: float):
    assert 0. <= score_threshold <= 1.
    for cross_corr, ids in create_pairwise_structs_in_db(database_name):
        if cross_corr.max_score >= score_threshold:
            yield cross_corr, ids


def generate_matching_pairs_from_npfile(filename: str, score_threshold: float, snr: float):
    assert 0. <= score_threshold <= 1.
    for cross_corr, ids in create_pairwise_structs_from_npfile(filename, snr):
        if cross_corr.max_score >= score_threshold:
            print(cross_corr.max_score, ids)
            yield cross_corr, ids


def generate_stretched_signals(zoom_range: (float, float), signal: np.ndarray, step=0.01):
    for zoom in create_range(zoom_range[0], zoom_range[1], step):
        if zoom != 1.:
            signal_mod = ndimage.zoom(signal, zoom)
            yield (signal_mod, signal_mod[::-1], zoom)
        else:
            signal_mod = np.array(signal)
            yield (signal_mod, signal_mod[::-1], zoom)


def max_function(_list):
    return _list[0]


def write_correlation_data_from_stream(correlation_stream, filename):
    corrs = list(correlation_stream)
    np.save(filename, corrs)
    print(len(corrs))


def npfile_to_corr_pipeline(input_filename: str, output_filename: str, snr: float, score: float):
    assert output_filename.endswith(".npy") and input_filename.endswith(".npy")
    corr_generator = generate_matching_pairs_from_npfile(input_filename, score, snr)
    write_correlation_data_from_stream(corr_generator, output_filename)


def subsample_from_npfile_and_save(input_filename: str, output_filename: str, ratio: float):
    assert 0. <= ratio <= 1.
    list_of_signals = ms.load_molecules_from_npfile(input_filename)
    np.save(output_filename, np.random.choice(list_of_signals, int(len(list_of_signals) * ratio), replace=False))


def load_corr_from_file(filename: str):
    assert filename.endswith("corr.npy")
    return np.load(filename)



