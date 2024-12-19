from torchtyping import TensorType as _Tensor
import torch
import torch.nn as nn
import pytorch_lightning as pl

from hicdifflib.utils import bp
from HiCDiffusion.hicdiffusion_encoder_decoder_model import HiCDiffusionEncoderDecoder
from HiCDiffusion.denoise_model import UnetConditional, GaussianDiffusionConditional


HICDIFFUSION_WINDOW_BP = bp(2_097_152)
HICDIFFUSION_WINDOW_SIZE = int(HICDIFFUSION_WINDOW_BP)
HICDIFFUSION_OUTPUT_CHANNELS = 512
HICDIFFUSION_OUTPUT_SIZE = 256


class HiCDiffusion(pl.LightningModule):
    def __init__(self, encoder_decoder_model: str):
        super().__init__()
        self.encoder_decoder = HiCDiffusionEncoderDecoder.load_from_checkpoint(encoder_decoder_model)
        self.encoder_decoder.freeze()
        self.encoder_decoder.eval()
        self.model = UnetConditional(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True,
            channels=1
        )
        self.diffusion = GaussianDiffusionConditional(
            self.model,
            image_size = 256,
            timesteps = 10,
            sampling_timesteps = 10
        )


class ResidualConv2d(nn.Module):

    def __init__(self, hidden_in, hidden_out, kernel, padding, dilation):
        super().__init__()
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
        super().__init__()
        self.reduce = reduce_layer
        self.model = (
            HiCDiffusionEncoderDecoder.load_from_checkpoint(checkpoint)
            if checkpoint else
            HiCDiffusionEncoderDecoder(None, None, None)
        )
        del self.model.reduce_layer
    
    def forward(
        self, 
        inputs: _Tensor['batch', 'onehot', 'sequence'], 
        mask: _Tensor['batch', 1, HICDIFFUSION_OUTPUT_SIZE, HICDIFFUSION_OUTPUT_SIZE] | None = None
    ) -> _Tensor['batch', 'hidden']:
        x = self.model.encoder(inputs)
        x = self.model.decoder(x)
        if mask is not None:
            x = torch.cat([x, mask], dim=1)
        x = self.reduce(x)
        return x
