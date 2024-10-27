import re
from os import PathLike
from math import floor, ceil
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torchtyping import TensorType as _Tensor


_unwanted_chars = re.compile(r'[^ACTG]')


def sequence_to_onehot(sequence: str) -> _Tensor['nucleotide', 'sequence']:
    sequence = list(re.sub(_unwanted_chars, 'N', sequence.upper()))
    codes = ['ACTGN'.index(char) for char in sequence]
    onehot = F.one_hot(torch.tensor(codes), 5).to(torch.float32)
    return torch.transpose(onehot, 0, 1)


def sequences_mask(
        n: int,
        start_l: int,
        end_l: int, 
        start_r: int, 
        end_r: int, 
        size: int
    ) -> _Tensor[1, 1, 'size', 'size']:
        mask = torch.zeros([1, 1, size, size], dtype=torch.float32)
        start, end = floor(start_l / n * (size-1)), ceil(end_l / n * (size-1))
        mask[:, :, start: end, :] = 1
        start, end = floor(start_r / n * (size-1)), ceil(end_r / n * (size-1))
        mask[:, :, :, start: end] = 1
        return mask



def read_paired_ends(
    path: PathLike,
    compression: str | None = None,
    header: bool = None,
    sep: str = '\t',
    filter_query: str = None,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    read_args = {
        'compression': compression or {
            '.gz': 'gzip'
        }.get(path.suffix),
        'sep': sep,
    }
    if not header:
        read_args['header'] = None
    pairs = pd.read_csv(path, **read_args)

    pairs.columns = [
        'chr_l', 'start_l', 'end_l', 'chr_r', 'start_r', 'end_r',
        *(extra_columns or [])
    ]
    pairs['len_l'] = (pairs.end_l - pairs.start_l).map(bp)
    pairs['len_r'] = (pairs.end_r - pairs.start_r).map(bp)
    pairs['len_full'] = (pairs.end_r - pairs.start_l).map(bp)

    if filter_query:
        pairs = pairs.query(filter_query).reset_index(drop=True)
    return pairs


def _reduce_to_unit(num):
    for unit in ("", "k", "M"):
        if abs(num) < 1000:
            return num, unit
        num /= 1000
    return num, 'G'


def basepairs_fmt(num):
    suffix = "bp"
    num, unit = _reduce_to_unit(num)
    if num.is_integer():
        return f"{int(num)}{unit}{suffix}"
    return f"{num:.1f}{unit}{suffix}"

class bp:
    def __init__(self, bp, unit=''):
        if isinstance(bp, str):
            suffix = bp[-3:]
            unit = suffix[0]
            if not suffix[1:] == 'bp':
                raise ValueError
            bp = bp[:-3]

        self._bp = float(bp)
        if unit:
            self._bp *= {'k': 1000, 'M': 1_000_000, 'G': 1_000_000_000}[unit]

    def __str__(self):
        return basepairs_fmt(self._bp)

    def __int__(self):
        return int(self._bp)

    def __float__(self):
        return float(self._bp)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __eq__(self, other):
        return int(self) == int(other)

    def __ne__(self, other):
        return int(self) != int(other)


def bint(*args, **kwargs):
    return int(bp(*args, **kwargs))


def bstr(*args, **kwargs):
    return str(bp(*args, **kwargs))

