from OptiMap import *
from OptiScan import database as db
from OptiMap.correlation_struct import EPSILON
"""
The class here is intended to store single OM molecule attributes.
"""


class Molecule:
    """
    Each molecule is initiated and explained with this class.
    """
    def __init__(self, nick_signal: np.ndarray, backbone_signal: np.ndarray, snr: float,
                 signal_filter=lambda x, y: (x, y), nick_filter=lambda x: True, log_snr=1.3):
        assert nick_signal.shape == backbone_signal.shape
        self.nick_signal = nick_signal
        self.backbone_signal = backbone_signal
        self.log_signal = None
        self.record_log_signal(log_snr)

        self.quality = True
        self.median = float(np.median(nick_signal))
        signal_info = utils.get_bnx_info_with_upsampling_with_forced_median(nick_signal, snr, self.median, 10, 1)
        self.nick_coordinates = signal_info["nick_distances"]
        self.nick_snrs = signal_info["nick_snrs"]

        self.filter_signal_pairs(func=signal_filter)
        self.check_nick_quality(func=nick_filter)

        try:
            assert self.nick_signal.shape[0] == self.backbone_signal.shape[0]
        except AssertionError:
            self.quality = False

    def check_nick_quality(self, func=lambda x: x):
        self.quality = func(self.nick_coordinates)

    def filter_signal_pairs(self, func=lambda x, y: (x, y)):
        self.nick_signal, self.backbone_signal = func(self.nick_signal, self.backbone_signal)

    def record_log_signal(self, snr: float):
        log_mol = np.log1p(self.nick_signal) + EPSILON
        median = np.median(self.nick_signal)
        self.log_signal = np.where(log_mol > snr*median, log_mol, EPSILON)


class MoleculeNoBack(Molecule):
    def __init__(self, nick_signal: np.ndarray, snr: float,
                 signal_filter=lambda x, y: (x, y), nick_filter=lambda x: True):
        backbone_signal = np.zeros(nick_signal.shape, dtype=nick_signal.dtype)
        Molecule.__init__(self, nick_signal, backbone_signal, snr, signal_filter=signal_filter, nick_filter=nick_filter)


def load_molecules_from_npfile(filename: str) -> [np.ndarray]:
    return np.load(filename, encoding="bytes")


def yield_molecules_from_saved_np(filename: str, snr: float):
    for nick_signal in load_molecules_from_npfile(filename):
        try:
            assert type(nick_signal) == np.ndarray
        except AssertionError:
            continue
        molecule = MoleculeNoBack(nick_signal, snr, signal_filter=delete_short, nick_filter=delete_low_nick_count)
        if molecule.quality:
            yield molecule


def yield_molecules_from_signal_list(signal_list, snr: float):
    for nick_signal in signal_list:
        try:
            assert type(nick_signal) == np.ndarray
        except AssertionError:
            continue
        molecule = MoleculeNoBack(nick_signal, snr, signal_filter=delete_short, nick_filter=delete_low_nick_count)
        if molecule.quality:
            yield molecule


def delete_short(nick_signal, backbone_signal, length=275):
    if nick_signal.shape[0] <= length:
        return np.zeros(1), backbone_signal
    else:
        return nick_signal, backbone_signal


def delete_low_nick_count(nick_coordinates, nick_count=5):
    if len(nick_coordinates) <= nick_count:
        return False
    else:
        return True


def yield_molecules_from_stream(molecule_stream, snr: float):
    for nick_signal, backbone_signal in molecule_stream:
        molecule = Molecule(nick_signal, backbone_signal, snr,
                            signal_filter=delete_short, nick_filter=delete_low_nick_count)
        if molecule.quality:
            yield molecule


def create_molecule_stream_from_runs(database_name: str, run_id: str, snr: float):
    conn = database.MoleculeConnector(database_name)
    return yield_molecules_from_stream(conn.yield_molecule_signals_in_run(run_id), snr)


def create_molecule_stream_from_db(database_name: str, snr: float):
    conn = database.MoleculeConnector(database_name)
    molecule_stream = conn.yield_molecule_signals_in_all_runs()
    return yield_molecules_from_stream(molecule_stream, snr)
