import re

from torchtyping import TensorType as _Tensor
import torch
import torch.nn.functional as F

from hicdifflib.utils import bp
from HiCDiffusion.hicdiffusion_encoder_decoder_model import HiCDiffusionEncoderDecoder


HICDIFFUSION_WINDOW_BP = bp(2_097_152)
HICDIFFUSION_WINDOW_SIZE = int(HICDIFFUSION_WINDOW_BP)
