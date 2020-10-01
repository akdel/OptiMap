import numpy as np
from dataclasses import dataclass
import typing as ty
import itertools
from OptiMap import molecule_struct as ms


def generate_bnx_arrays(bnx_filename: str) -> ty.Iterator[dict]:
    lines: ty.List[str] = list()
    for k, g in itertools.groupby((x for x in open(bnx_filename, "r")), lambda x: x.split("\t")[0]):
        if k.startswith("#"):
            continue
        else:
            if len(lines) == 4:
                yield {
                    "info": [x.strip() for x in lines[0].split("\t")],
                    "labels": [float(x) for x in lines[1].split()[1:]],
                    "label_snr": [float(x) for x in lines[2].split()[1:]],
                    "raw_intensities": [float(x) for x in lines[3].split()[1:]],
                    "signal": None
                }
                lines: ty.List[str] = [list(g)[0]]
            else:
                lines.append(list(g)[0])

@dataclass
class SquareWave:
    labels: ty.List[int]
    length: int
    wave_width: int
    wave: np.ndarray
    zoom: int
    idx: int

    @classmethod
    def from_bnx_line(cls, bnx_array_entry: dict, reverse=False,
                      zoom_factor: int = 500,
                      snr: float = 3.5,
                      width: int = 10) -> "SquareWave":
        index: int = int(bnx_array_entry["info"][1])
        length: float = float(bnx_array_entry["info"][2])
        labels: ty.List[float] = list((np.array(bnx_array_entry["labels"][:-1])).astype(int)[
            np.array(bnx_array_entry["label_snr"]) >= snr])
        if reverse:
            index *= -1
        return cls.from_labels(labels,
                               length,
                               zoom=zoom_factor,
                               reverse=reverse,
                               width=width,
                               idx=index)

    @classmethod
    def from_optimap_molecule(cls,
                              optimap_molecule: ms.MoleculeNoBack,
                              molecule_id: int,
                              reverse: bool = False,
                              zoom: int = 1,
                              log: bool = False) -> 'SquareWave':
        labels: ty.List[int] = [int(x) for x in optimap_molecule.nick_coordinates]
        if reverse:
            labels: ty.List[int] = [(optimap_molecule.nick_signal.shape[0] - x) for x in labels]
        if log:
            wave: np.ndarray = optimap_molecule.log_signal
            wave[wave < 1] = 0
            wave[wave >= 1] = 1
        else:
            wave: np.ndarray = optimap_molecule.nick_signal
        return cls(labels, wave.shape[0], -1, wave if not reverse else wave[::-1],
                   zoom, molecule_id if not reverse else -molecule_id)

    @classmethod
    def from_labels(cls,
                    labels: ty.List[float],
                    length: float,
                    zoom: int = 1,
                    width: int = 10,
                    reverse: bool = False,
                    idx: int = 0) -> 'SquareWave':
        if reverse:
            labels: ty.List[float] = [(length - x) for x in labels]
        if zoom < 1:
            zoom: int = 1
        assert length >= max(labels)
        half_width: int = width//2
        labels: ty.List[int] = [int(x // zoom) for x in labels]
        length: int = int(length // zoom)
        wave: np.ndarray = np.zeros(length, dtype="int8")
        for label in labels:
            start, end = max(0, label - half_width), min(length, label + half_width)
            wave[start: end] = 1
        assert len(labels) > 1
        return cls(labels, length, width, wave, zoom, idx)

    def reform_wave_from_labels_with_width(self, width: int = 10) -> "SquareWave":
        return self.from_labels(self.labels, self.length, 1, width, False, self.idx)

    def segment(self, length: int = 150000) -> np.ndarray:
        segments: ty.List[ty.List[int]] = list()
        length //= self.zoom
        for label in self.labels:
            if (self.length - label) >= length:
                segments.append(list(self.wave[label: label + length]))
            else:
                continue
        return np.array(segments).astype("uint8")

    def compress(self, length: int = 150000, nbits: int = 32, limit: ty.Tuple[float, float] = (0.25, 0.75)):
        segments: np.ndarray = self.segment(length)
        if not len(segments):
            return None
        compressed_segments: np.ndarray = np.zeros((segments.shape[0], nbits), dtype=np.uint8)
        ranges = np.round(np.linspace(0, segments.shape[1], nbits), 0).astype(int)
        interval: int = ranges[1]
        limit_lower, limit_upper = int(nbits * limit[0]), int(nbits * limit[1])
        for i, segment in enumerate(segments):
            for j, adjusted_j in enumerate(ranges):
                compressed_segments[i, j] = np.sum(segment[adjusted_j: adjusted_j + interval]) >= (interval // 2)
        compressed_segments: np.ndarray = np.array([x for x in compressed_segments if limit_lower <= np.sum(x) <= limit_upper])
        if not len(compressed_segments):
            return None
        return np.packbits(compressed_segments).view(f"uint{nbits}").flatten()


if __name__ == "__main__":
    length_ = 350
    labels_ = list(np.sort(np.random.randint(0, length_, 25)).astype(float))
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    sq = SquareWave.from_labels(labels_, length_, width=8)
    segs = sq.compress(75)
    for seg in np.unpackbits(segs.view("uint8")).reshape((segs.shape[0], -1)):
        plt.plot(sq.wave)
        plt.show()
        plt.plot(seg)
        plt.show()
