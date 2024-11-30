import os
import logging
from pathlib import Path

import wandb
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers.models.bert.configuration_bert import BertConfig

from hicdifflib.data import PairedEndsDataset, DataConfig, WandbArtifact
from hicdifflib.dnabert import PairEncoderConfig, PairEncoderForClassification
from hicdifflib.hicdiffusion import HICDIFFUSION_WINDOW_SIZE
from hicdifflib.nn import PairedEndsCollatorWithPadding, make_weights_for_balanced_classes, CustomTrainer


logger = logging.getLogger(__name__)
run_config = dict(
    save_code=True, 
    job_type='train',
    project='HiCDiffusionLooping',
    name="train_dnabert",
    group="dnabert",
)

def main(run):
    test_chroms = ['chr14']
    eval_chroms = ['chr15']
    # train_chroms = ['chr1']
    train_chroms = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in (test_chroms+eval_chroms)]
    training_args = TrainingArguments(
        output_dir="/mnt/evafs/scratch/shared/imialeshka/hicdata/test3",
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
        run_name='dnabert',
        disable_tqdm=False,
        dataloader_num_workers=8,
        metric_for_best_model='average_precision',
        greater_is_better=True,
    )
    
    for key, value in os.environ.items():
        if key.startswith('SLURM_'):
            run.config[key] = value
    
    data_config = DataConfig(data_root='/mnt/evafs/scratch/shared/imialeshka/hicdata/')
    config_kwargs = dict(
        anchor_encoder_shared=True,
        hicdiffusion_frozen=False,
    )
    config = PairEncoderConfig(
        **config_kwargs,
        hicdiffusion_checkpoint=str(WandbArtifact('hicdiffusion_encoder_decoder:v0', data_config))
    )
    for key, value in config_kwargs.items():
        run.config[key] = value

    tokenizer = AutoTokenizer.from_pretrained("m10an/DNABERT-S", trust_remote_code=True)
    model = PairEncoderForClassification(config).cuda()

    common = dict(
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
    for key, value in common.items():
        if isinstance(value, (int, float, str)):
            run.config[key] = value    
    
    train_dataset = PairedEndsDataset(
        chroms=train_chroms,
        **common,
    )
    eval_dataset = PairedEndsDataset(
        chroms=eval_chroms,
        center_context=True,
        **common
    )
    
    trainer = CustomTrainer(
        model=model,
        data_collator=PairedEndsCollatorWithPadding(tokenizer),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # train_dataset=torch.utils.data.Subset(train_dataset, list(range(10))),
        # eval_dataset=torch.utils.data.Subset(eval_dataset, list(range(10))),
        compute_metrics=compute_metrics
    )

    res = trainer.train()


if __name__ == '__main__':
    with wandb.init(**run_config) as run:
        main(run)
