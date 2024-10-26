from torchtyping import TensorType as _Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from hicdifflib.utils import bp
from HiCDiffusion.hicdiffusion_encoder_decoder_model import HiCDiffusionEncoderDecoder


HICDIFFUSION_WINDOW_BP = bp(2_097_152)
HICDIFFUSION_WINDOW_SIZE = int(HICDIFFUSION_WINDOW_BP)
    

class ResidualConv2d(nn.Module):

    def __init__(self, hidden_in, hidden_out, kernel, padding, dilation):
        super(ResidualConv2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(hidden_in, hidden_out, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm2d(hidden_out),
            nn.ReLU(),
            nn.Conv2d(hidden_out, hidden_out, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm2d(hidden_out)
        )
        self.relu = nn.ReLU()
        self.downscale = nn.Sequential(
            nn.Conv2d(hidden_in, hidden_out, kernel, padding=padding)
        )
    
    def forward(self, x):
        residual = self.downscale(x)
        output = self.main(x)
        return self.relu(output+residual)    


class HiCDiffusionContextEncoder(nn.Module):
    def __init__(self, reduce_layer: nn.Module, checkpoint: str | None = None) -> None:
        self.reduce = reduce_layer
        self.model = (
            HiCDiffusionEncoderDecoder.load_from_checkpoint(checkpoint)
            if checkpoint else
            HiCDiffusionEncoderDecoder(None, None, None)
        )
        del self.model.reduce_layer
        super().__init__()
    
    def forward(
        self, 
        inputs: _Tensor['batch', 'onehot', 'sequence'], 
        mask: _Tensor['batch', 1, 256, 256]
    ) -> _Tensor['batch', 'hidden']:
        x = self.model.encoder(inputs)
        x = self.model.decoder(x)
        x = torch.cat([x, mask], dim=1)
        x = self.reduce(x)
        return x
