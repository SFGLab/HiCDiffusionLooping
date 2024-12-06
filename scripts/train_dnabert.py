import os
import logging

import wandb
import torch.nn as nn
from torch.utils.data import Subset
from transformers import AutoTokenizer
from transformers import TrainingArguments

from hicdifflib.data.base import DataConfig, WandbArtifact
from hicdifflib.data.collator import PairedEndsCollatorWithPadding
from hicdifflib.data.dataset import PairedEndsDataset
from hicdifflib.dnabert import PairEncoderConfig, PairEncoderForClassification
from hicdifflib.metrics import compute_metrics
from hicdifflib.trainer import BalancedTrainer


logger = logging.getLogger(__name__)


DEBUG=False
RUN_NAME='dnabert'

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
        lr_scheduler_type='reduce_lr_on_plateau',
        lr_scheduler_kwargs=dict(mode='max', factor=0.5, patience=1),
        warmup_steps=10000,
        num_train_epochs=10,
        evaluation_strategy='steps',
        eval_steps=10000,
        eval_delay=0,
        save_strategy='steps',
        save_steps=10000,
        logging_strategy='steps',
        logging_steps=10,
        auto_find_batch_size=True,
        # per_device_eval_batch_size=8,
        # per_device_train_batch_size=8,
        run_name=RUN_NAME,
        disable_tqdm=False,
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
        min_length_match=0.95,
        progress_bar=False,
        positive_min_pet_counts=3,
        negative_max_pet_counts=0,
        max_anchor_length=510,
    )
    config_kwargs = dict(
        anchor_encoder_shared=True,
        hicdiffusion_frozen=True,
    )
    run.config.update(config_kwargs)
    
    for key, value in os.environ.items():
        if key.startswith('SLURM_'):
            run.config[key] = value
    
    for key, value in dataset_kwargs.items():
        if isinstance(value, (int, float, str)):
            run.config[key] = value    
    
    train_dataset = PairedEndsDataset(
        chroms=train_chroms,
        **dataset_kwargs,
    )
    eval_dataset = PairedEndsDataset(
        chroms=eval_chroms,
        center_context=True,
        **dataset_kwargs
    )
    config = PairEncoderConfig(
        **config_kwargs,
        hicdiffusion_checkpoint=str(WandbArtifact('hicdiffusion_encoder_decoder:v0', data_config))
    )
    model = PairEncoderForClassification(config).cuda()
    trainer = BalancedTrainer(
        model=model,
        data_collator=PairedEndsCollatorWithPadding(tokenizer),
        args=training_args,
        train_dataset=train_dataset if not DEBUG else Subset(train_dataset, list(range(10))),
        eval_dataset=eval_dataset if not DEBUG else Subset(eval_dataset, list(range(10))),
        compute_metrics=compute_metrics
    )
    run.config['trainer_type'] = str(trainer.__class__.__name__)
    run.config['loss_type'] = str(nn.BCEWithLogitsLoss.__class__.__name__)
    
    trainer.train()


if __name__ == '__main__':
    with wandb.init(**run_config) as run:
        main(run)
