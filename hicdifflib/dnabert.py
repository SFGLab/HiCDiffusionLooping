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
        hicdiffusion_checkpoint: str | None = None,
        hicdiffusion_frozen: bool = True,
        anchor_encoder: str = "m10an/DNABERT-S",
        anchor_encoder_shared: bool = False,
        anchor_encoder_frozen: bool = True,
        hidden_size: int = 768,
        hidden_dropout: float = 0.1,
        classifier_dropout: float | None = None,
        **kwargs,
    ):
        self.anchor_encoder = anchor_encoder
        self.anchor_encoder_shared = anchor_encoder_shared
        self.anchor_encoder_frozen = anchor_encoder_frozen
        self.hicdiffusion_checkpoint = hicdiffusion_checkpoint
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = (
            hidden_dropout
            if classifier_dropout is None else
            classifier_dropout
        )
        self.hicdiffusion_frozen = hicdiffusion_frozen
        super().__init__(**kwargs)


class MeanPoolingEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, *args, **kwargs) -> _Tensor['batch', 'features']:
        encoded, pooled = self.encoder(*args, **kwargs)
        return torch.mean(encoded, dim=1)


class PairEncoderModel(PreTrainedModel):
    config_class = PairEncoderConfig

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _anchor_encoder(self, config):
        model = AutoModel.from_pretrained(config.anchor_encoder, trust_remote_code=True)
        if config.anchor_encoder_frozen:
            self._freeze_model(model)
        return MeanPoolingEncoder(model)


    def __init__(self, config: PairEncoderConfig):
        super().__init__(config)
        self.left_model = self._anchor_encoder(config)
        self.right_model = (
            self.left_model if config.anchor_encoder_shared else self._anchor_encoder(config)
        )
        self.context_model = HiCDiffusionContextEncoder(
            reduce_layer=nn.Sequential(
                ResidualConv2d(HICDIFFUSION_OUTPUT_CHANNELS + 1, 256, 3, 1, 1), 
                ResidualConv2d(256, 128, 3, 1, 1), 
                ResidualConv2d(128, 64, 3, 1, 1), 
                ResidualConv2d(64, 32, 3, 1, 1), 
                ResidualConv2d(32, 16, 3, 1, 1), 
                ResidualConv2d(16, 8, 3, 1, 1), 
                ResidualConv2d(8, 1, 3, 1, 1),
                nn.Flatten(),
                nn.Dropout(config.hidden_dropout),
                nn.Linear(HICDIFFUSION_OUTPUT_SIZE * HICDIFFUSION_OUTPUT_SIZE, config.hidden_size),
                nn.ReLU(),
            ),
            checkpoint=config.hicdiffusion_checkpoint
        )
        if config.hicdiffusion_frozen:
            self._freeze_model(self.context_model.model.encoder)
            self._freeze_model(self.context_model.model.decoder)
        self.final = nn.Sequential(nn.Linear(config.hidden_size*3, config.hidden_size), nn.ReLU())

    def forward(
        self, 
        left_input_ids: _Tensor['batch', 'sequence'],
        right_input_ids: _Tensor['batch', 'sequence'],
        context_sequence: _Tensor['batch', 'onehot', 'sequence'],
        context_mask: _Tensor['batch', 'w', 'h'],
        left_attention_mask: _Tensor['batch', 'sequence'] | None = None,
        right_attention_mask: _Tensor['batch', 'sequence'] | None = None,
    ):
        left_hidden = self.left_model(
            input_ids=left_input_ids,
            attention_mask=left_attention_mask,
        )
        right_hidden = self.right_model(
            input_ids=right_input_ids,
            attention_mask=right_attention_mask,
        )
        context_hidden = self.context_model(context_sequence, context_mask)
        x = torch.cat([left_hidden, right_hidden, context_hidden], dim=1)
        x = self.final(x)
        return x


class PairEncoderForClassification(PreTrainedModel):
    config_class = PairEncoderConfig

    def __init__(self, config: PairEncoderConfig):
        super().__init__(config)
        self.encoder = PairEncoderModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self, 
        left_input_ids: _Tensor['batch', 'sequence'],
        right_input_ids: _Tensor['batch', 'sequence'],
        context_sequence: _Tensor['batch', 'onehot', 'sequence'],
        context_mask: _Tensor['batch', 'w', 'h'],
        labels: _Tensor['batch', 'label'] | None = None,
        left_attention_mask: _Tensor['batch', 'sequence'] | None = None,
        right_attention_mask: _Tensor['batch', 'sequence'] | None = None,
    ):
        x = self.encoder(
            left_input_ids=left_input_ids,
            right_input_ids=right_input_ids,
            left_attention_mask=left_attention_mask,
            right_attention_mask=right_attention_mask,
            context_sequence=context_sequence,
            context_mask=context_mask,
        )
        x = self.dropout(x)
        logits = self.classifier(x)
        
        if labels is None:
            return (logits, )
        
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, logits)
