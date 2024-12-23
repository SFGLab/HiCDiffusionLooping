import os
import sys
import logging

import wandb
import torch.nn as nn
from torch.utils.data import Subset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

from hicdifflib.data.base import DataConfig, WandbArtifact
from hicdifflib.data.collator import PairedEndsCollatorWithPadding
from hicdifflib.data.dataset import PairedEndsDataset
from hicdifflib.v2 import PairEncoderConfig2 as PairEncoderConfig
from hicdifflib.v2 import PairEncoderForClassification
from hicdifflib.metrics import compute_metrics
from hicdifflib.trainer import BalancedTrainer


logger = logging.getLogger(__name__)


DEBUG=False
RUN_NAME=sys.argv[1]

if DEBUG:
    RUN_NAME += '_debug'

run_config = dict(
    save_code=True, 
    job_type='train' if not DEBUG else 'debug',
    project='HiCDiffusionLooping',
    name=f"train_{RUN_NAME}",
    group=RUN_NAME,
)

def main(run):
    test_chroms = ['chr14']
    eval_chroms = ['chr15']
    train_chroms = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in (test_chroms+eval_chroms)]
    if DEBUG:
        train_chroms = train_chroms[:1]
    
    tokenizer = AutoTokenizer.from_pretrained("m10an/DNABERT-S", trust_remote_code=True)
    data_config = DataConfig(data_root='/mnt/evafs/scratch/shared/imialeshka/hicdata/')
    training_args = TrainingArguments(
        output_dir=f"/mnt/evafs/scratch/shared/imialeshka/hicdata/{RUN_NAME}",
        warmup_steps=10_000,
        num_train_epochs=20,
        evaluation_strategy='steps',
        eval_steps=10_000,
        eval_delay=0,
        save_total_limit=3,
        save_strategy='steps',
        save_steps=10_000,
        logging_strategy='steps',
        logging_steps=10,
        # auto_find_batch_size=True,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        run_name=RUN_NAME,
        disable_tqdm=True,
        dataloader_num_workers=8,
        metric_for_best_model='average_precision',
        greater_is_better=True,
    )

    dataset_kwargs = dict(
        pairs=WandbArtifact('pet_pairs.csv:v1', data_config, run).path,
        sequences=[
            WandbArtifact('GRCh38-reference-genome:v0', data_config, run).path,
            WandbArtifact('4DNFI1GNQM8L.delly.vcf.fa:v0', data_config, run).path,
            WandbArtifact('4DNFI2OEE66L.delly.vcf.fa:v0', data_config, run).path,
        ],
        tokenizer=tokenizer,
        progress_bar=False,
        positive_min_pet_counts=3,
        negative_max_pet_counts=2,
        max_anchor_length=510,
    )
    config_kwargs = dict(
        encoder_decoder_checkpoint=str(WandbArtifact('hicdiffusion_encoder_decoder:v0', data_config)),
        encoder_decoder_frozen=True,
        diffusion_checkpoint=str(WandbArtifact('hicdiffusion:v0', data_config)),
        diffusion_flash_attn=False,
        diffusion_frozen=True,
        hicdiffusion_mask=True,
        anchor_encoder="m10an/DNABERT-S",
        anchor_encoder_shared=True,
    )
    run.config.update(config_kwargs)
    
    for key, value in os.environ.items():
        if key.startswith('SLURM_'):
            run.config[key] = value
    
    for key, value in dataset_kwargs.items():
        if isinstance(value, (int, float, str)):
            run.config[key] = value

    run.config['min_length_match'] = run.config['train_min_length_match'] = 0.2
    run.config['eval_length_match'] = 0.95
    
    train_dataset = PairedEndsDataset(
        chroms=train_chroms,
        min_length_match=run.config['train_min_length_match'],
        **dataset_kwargs,
    )
    eval_dataset = PairedEndsDataset(
        chroms=eval_chroms,
        center_context=True,
        min_length_match=run.config['eval_length_match'],
        **dataset_kwargs
    )
    config = PairEncoderConfig(**config_kwargs)
    model = PairEncoderForClassification(config).cuda()
    
    trainer = (Trainer if DEBUG else BalancedTrainer)(
        model=model,
        data_collator=PairedEndsCollatorWithPadding(tokenizer),
        args=training_args,
        train_dataset=train_dataset if not DEBUG else Subset(train_dataset, list(range(10))),
        eval_dataset=eval_dataset if not DEBUG else Subset(eval_dataset, list(range(10))),
        compute_metrics=compute_metrics
    )
    run.config['trainer_type'] = str(trainer.__class__.__name__)
    run.config['loss_type'] = str(nn.BCEWithLogitsLoss.__name__)
    
    trainer.train()


if __name__ == '__main__':
    with wandb.init(**run_config) as run:
        main(run)
