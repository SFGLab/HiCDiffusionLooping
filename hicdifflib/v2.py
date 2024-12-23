import torch
import torch.nn as nn
from torchtyping import TensorType as _Tensor
from transformers import PretrainedConfig, PreTrainedModel, AutoModel

from hicdifflib.hicdiffusion import HICDIFFUSION_OUTPUT_CHANNELS
from hicdifflib.hicdiffusion import HICDIFFUSION_OUTPUT_SIZE
from hicdifflib.hicdiffusion import ResidualConv2d, HiCDiffusion


class PairEncoderConfig2(PretrainedConfig):
    model_type = "pairencoder2"

    def __init__(
        self,
        encoder_decoder_checkpoint: str | None = None,
        encoder_decoder_frozen: bool = True,
        diffusion_checkpoint: str | None = None,
        diffusion_flash_attn: bool = False,
        diffusion_frozen: bool = True,
        hicdiffusion_mask: bool = True,
        anchor_encoder: str | None = "m10an/DNABERT-S",
        anchor_encoder_shared: bool = True,
        hidden_size: int = 768,
        hidden_dropout: float = 0.1,
        classifier_dropout: float | None = None,
        **kwargs,
    ):
        self.encoder_decoder_checkpoint = encoder_decoder_checkpoint
        self.encoder_decoder_frozen = encoder_decoder_frozen
        self.diffusion_checkpoint = diffusion_checkpoint
        self.diffusion_flash_attn = diffusion_flash_attn
        self.diffusion_frozen = diffusion_frozen
        self.hicdiffusion_mask = hicdiffusion_mask
        self.anchor_encoder = anchor_encoder
        self.anchor_encoder_shared = anchor_encoder_shared
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = (
            hidden_dropout
            if classifier_dropout is None else
            classifier_dropout
        )
        super().__init__(**kwargs)


class MeanPoolingEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, *args, **kwargs) -> _Tensor['batch', 'features']:
        encoded, pooled = self.encoder(*args, **kwargs)
        return torch.mean(encoded, dim=1)


class PairEncoderModel(PreTrainedModel):
    config_class = PairEncoderConfig2

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _anchor_encoder(self, config):
        if config.anchor_encoder is None:
            return None
        model = AutoModel.from_pretrained(config.anchor_encoder, trust_remote_code=True)
        if config.anchor_encoder_frozen:
            self._freeze_model(model)
        return MeanPoolingEncoder(model)
    
    def _context_encoder(self, config: PairEncoderConfig2):        
        model = HiCDiffusion.from_checkpoint(
            config.diffusion_checkpoint,
            encoder_decoder_model=config.encoder_decoder_checkpoint, 
            flash_attn=config.flash_attn,
        )
        
        if config.encoder_decoder_frozen:
            self._freeze_model(model.encoder_decoder)
        
        if config.diffusion_frozen:
            self._freeze_model(model.model)
            self._freeze_model(model.diffusion)
            
        return model

    def __init__(self, config: PairEncoderConfig2):
        super().__init__(config)
        self.config: PairEncoderConfig2
        self.left_model = self._anchor_encoder(config)
        self.right_model = (
            self.left_model if config.anchor_encoder_shared else self._anchor_encoder(config)
        )
        self.context_model = self._context_encoder(config)
        self.use_context_mask = self.config.hicdiffusion_mask
        self.use_context_features = self.context_model is not None
        self.use_anchor_features = self.left_model is not None
        self.context_reduce = nn.Sequential(
            ResidualConv2d(
                HICDIFFUSION_OUTPUT_CHANNELS     # y_cond channels
                + self.config.hicdiffusion_mask  # mask channel
                + 1,                             # y_pred channel
                256, 3, 1, 1
            ),
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
        )
        self.final = nn.Sequential(
            nn.Linear(
                in_features=config.hidden_size * (2 * self.use_anchor_features + self.use_context_features), 
                out_features=config.hidden_size
            ),
            nn.ReLU()
        )

    def forward(
        self, 
        left_sequence: _Tensor['batch', 'sequence'],
        right_sequence: _Tensor['batch', 'sequence'],
        context_sequence: _Tensor['batch', 'onehot', 'sequence'],
        context_mask: _Tensor['batch', 'w', 'h'],
        left_attention_mask: _Tensor['batch', 'sequence'] | None = None,
        right_attention_mask: _Tensor['batch', 'sequence'] | None = None,
    ):
        features = []
        if self.use_anchor_features:
            left_hidden = self.left_model(
                input_ids=left_sequence,
                attention_mask=left_attention_mask,
            )
            right_hidden = self.right_model(
                input_ids=right_sequence,
                attention_mask=right_attention_mask,
            )
            features.extend([left_hidden, right_hidden])
        
        context = self.context_model(context_sequence)
        if self.use_context_mask:
            context.append(context_mask)
        features.append(self.context_reduce(context))
        
        x = torch.cat(features, dim=1)
        x = self.final(x)
        return x


class PairEncoderForClassification(PreTrainedModel):
    config_class = PairEncoderConfig2
    def __init__(self, config: PairEncoderConfig2):
        super().__init__(config)
        self.encoder = PairEncoderModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self, 
        left_sequence: _Tensor['batch', 'sequence'],
        right_sequence: _Tensor['batch', 'sequence'],
        context_sequence: _Tensor['batch', 'onehot', 'sequence'],
        context_mask: _Tensor['batch', 'w', 'h'],
        labels: _Tensor['batch', 'label'] | None = None,
        left_attention_mask: _Tensor['batch', 'sequence'] | None = None,
        right_attention_mask: _Tensor['batch', 'sequence'] | None = None,
    ):
        x = self.encoder(
            left_sequence=left_sequence,
            right_sequence=right_sequence,
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
