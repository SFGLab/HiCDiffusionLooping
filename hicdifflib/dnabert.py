import torch
import torch.nn as nn
from torchtyping import TensorType as _Tensor
from transformers import PretrainedConfig, PreTrainedModel, AutoModel

from hicdifflib.hicdiffusion import HICDIFFUSION_OUTPUT_CHANNELS
from hicdifflib.hicdiffusion import HICDIFFUSION_OUTPUT_SIZE
from hicdifflib.hicdiffusion import ResidualConv2d, HiCDiffusionContextEncoder


class PairEncoderConfig(PretrainedConfig):
    model_type = "pairencoder"

    def __init__(
        self,
        hicdiffusion_checkpoint: str,
        anchor_encoder: str = "m10an/DNABERT-S",
        anchor_encoder_shared: bool = False,
        hidden_size: int = 768,
        **kwargs,
    ):
        self.anchor_encoder = anchor_encoder
        self.anchor_encoder_shared = anchor_encoder_shared
        self.hicdiffusion_checkpoint = hicdiffusion_checkpoint
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class MeanPooling(nn.Module):
    def forward(self, x: _Tensor['batch', 'sequence', 'features']) -> _Tensor['batch', 'features']:
        return torch.mean(x, dim=1)


class PairEncoder(nn.Module):
    def __init__(
            self, 
            left_model: nn.Module, 
            right_model: nn.Module, 
            context_model: nn.Module,
            hidden_size: int
        ) -> None:
        super().__init__()
        self.left_model = left_model
        self.right_model = right_model
        self.context_model = context_model
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size*3, hidden_size)
    
    def forward(
        self, 
        left_ids: _Tensor['batch', 'sequence'],
        right_ids: _Tensor['batch', 'sequence'],
        context_sequence: _Tensor['batch', 'onehot', 'sequence'],
        context_mask: _Tensor['batch', 'w', 'h'],
    ):
        left_hidden = self.left_model(left_ids)
        right_hidden = self.right_model(right_ids)
        context_hidden = self.context_model(context_sequence, context_mask)
        x = torch.cat([left_hidden, right_hidden, context_hidden])
        x = self.fc(x)
        return x


class PairEncoderModel(PreTrainedModel):
    config_class = PairEncoderConfig

    def __init__(self, config: PairEncoderConfig):
        super().__init__(config)
        left_model = AutoModel.from_pretrained(config.anchor_encoder, trust_remote_code=True)
        right_model = (
            left_model 
            if self.anchor_encoder_shared else
            AutoModel.from_pretrained(config.anchor_encoder, trust_remote_code=True)
        )
        context_model = HiCDiffusionContextEncoder(
            reduce_layer=nn.Sequential(
                ResidualConv2d(HICDIFFUSION_OUTPUT_CHANNELS + 1, 256, 3, 1, 1), 
                ResidualConv2d(256, 128, 3, 1, 1), 
                ResidualConv2d(128, 64, 3, 1, 1), 
                ResidualConv2d(64, 32, 3, 1, 1), 
                ResidualConv2d(32, 16, 3, 1, 1), 
                ResidualConv2d(16, 8, 3, 1, 1), 
                ResidualConv2d(8, 1, 3, 1, 1),
                nn.Flatten(),
                nn.Linear(HICDIFFUSION_OUTPUT_SIZE * HICDIFFUSION_OUTPUT_SIZE, config.hidden_size),
                nn.ReLU(),
            ),
            checkpoint=config.hicdiffusion_checkpoint
        )
        self.model = PairEncoder(left_model, right_model, context_model, config.hidden_size)

    def forward(self, **inputs):
        return self.model(**inputs)
