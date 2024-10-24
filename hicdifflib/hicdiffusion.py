import re

from torchtyping import TensorType as _Tensor
import torch
import torch.nn.functional as F

from hicdifflib.utils import bp
from HiCDiffusion.hicdiffusion_encoder_decoder_model import HiCDiffusionEncoderDecoder


HICDIFFUSION_WINDOW_BP = bp(2_097_152)
HICDIFFUSION_WINDOW_SIZE = int(HICDIFFUSION_WINDOW_BP)

_unwanted_chars = re.compile(r'[^ACTG]')


def sequence_to_onehot(sequence: str) -> _Tensor['nucleotide', 'sequence']:
    sequence = list(re.sub(_unwanted_chars, 'N', sequence.upper()))
    codes = ['ACTGN'.index(char) for char in sequence]
    onehot = F.one_hot(torch.tensor(codes), 5).to(torch.float)
    return torch.transpose(onehot, 0, 1)
