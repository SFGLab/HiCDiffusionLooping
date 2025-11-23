import os
import sys
import pickle
import logging
import hashlib
from pathlib import Path

import pandas as pd
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Subset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from safetensors.torch import load_file

from hicdifflib.data.base import DataConfig, WandbArtifact
from hicdifflib.data.collator import PairedEndsCollatorWithPadding
from hicdifflib.data.dataset import PairedEndsDataset, ConcatPairedEndsDataset
from hicdifflib.dnabert import PairEncoderConfig2 as PairEncoderConfig
from hicdifflib.dnabert import PairEncoderForClassification
from hicdifflib.metrics import compute_metrics
from hicdifflib.trainer import BalancedTrainer


logger = logging.getLogger(__name__)

DEBUG = len(sys.argv) > 2
RUN_NAME=sys.argv[1]
USE_CACHE=False


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
    data_config = DataConfig(data_root='./data/')
    training_args = TrainingArguments(
        output_dir=f"./data/{RUN_NAME}",
        warmup_steps=5_000,
        num_train_epochs=20,
        evaluation_strategy='steps',
        eval_steps=10_000,
        eval_delay=0,
        eval_on_start=DEBUG,
        save_total_limit=3,
        save_strategy='steps',
        save_steps=10_000,
        save_safetensors=False, # doesnt work with shared hicdiffusion modules
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
        warmup_ratio=0,
    )

    run_config = dict(
        context_reduce_pretrained='encoder_pretrain/checkpoint-90000/pytorch_model.bin',
        hic=None,
        # hic=WandbArtifact('gm12878_4DNFIUEG39YZ:v0', data_config).path,
        progress_bar=True,
        tokenizer=tokenizer,
        positive_min_pet_counts=3,
        negative_max_pet_counts=2,
        max_anchor_length=510,
        reference=WandbArtifact('GRCh38-reference-genome:v0', data_config, run).path,
        gm12878_pairs=WandbArtifact('gm12878_pairs.csv:latest', data_config, run).path,
        gm12878_sequences=[
            WandbArtifact('4DNFI1GNQM8L.delly.vcf.fa:v0', data_config, run).path,
            WandbArtifact('4DNFI2OEE66L.delly.vcf.fa:v0', data_config, run).path,
        ],
        min_length_match=0.2,
        train_min_length_match=0.2,
        eval_length_match=0.95,
    )
    
    model_config = dict(
        encoder_decoder_checkpoint=str(WandbArtifact('hicdiffusion_encoder_decoder:v0', data_config)),
        encoder_decoder_frozen=True,
        diffusion_checkpoint=str(WandbArtifact('hicdiffusion:v0', data_config)),
        diffusion_flash_attn=False,
        diffusion_frozen=True,
        hicdiffusion_mask=True,
        hicdiffusion_y_cond=False,
        hicdiffusion_cnn='3d',
        anchor_encoder="m10an/DNABERT-S",
        anchor_encoder_shared=True,
    )
    run.config.update(model_config)
    model_config['hic'] = False
    
    for key, value in os.environ.items():
        if key.startswith('SLURM_'):
            run.config[key] = value
    
    for key, value in run_config.items():
        if isinstance(value, (int, float, str, Path)):
            run.config[key] = value

    def pairs_dataset(name, hash_kwargs, **kwargs):
        cache_dir = Path(data_config.data_root) / 'pairs_dataset_cache'
        cache_dir.mkdir(exist_ok=True)
    
        md5 = hashlib.md5(repr(hash_kwargs).encode()).hexdigest()
        pairs_path = cache_dir / f'{name}_pairs_{md5}.csv'
        vs_path = cache_dir / f'{name}_valid_sequences_{md5}.pkl'

        if USE_CACHE and pairs_path.exists() and vs_path.exists():
            print('use cached:', str(pairs_path), str(vs_path))
            return PairedEndsDataset(
                **hash_kwargs,
                **kwargs,
                pairs_df=pd.read_csv(pairs_path),
                valid_sequences=pickle.loads(vs_path.read_bytes())
            )

        dataset = PairedEndsDataset(
            **hash_kwargs,
            **kwargs
        )
        if USE_CACHE:
            dataset._pairs_df.to_csv(pairs_path, index=False)
            vs_path.write_bytes(pickle.dumps(dataset._valid_sequences))
        return dataset

    gm12878_kwargs = dict(
        pairs=run_config['gm12878_pairs'],
        sequences=[run_config['reference'], *run_config['gm12878_sequences']],
        tokenizer=tokenizer,
        positive_min_pet_counts=run_config['positive_min_pet_counts'],
        negative_max_pet_counts=run_config['negative_max_pet_counts'],
        max_anchor_length=run_config['max_anchor_length'],
    )

    train_dataset_gm12878 = pairs_dataset(
        name='train_gm12878', 
        hash_kwargs=dict(
            chroms=train_chroms,
            min_length_match=run_config['train_min_length_match'],
            **gm12878_kwargs
        ),
        progress_bar=run_config['progress_bar'],
        hic=run_config['hic'],
    )
    train_dataset = train_dataset_gm12878

    eval_dataset = pairs_dataset(
        name='eval_gm12878', 
        hash_kwargs=dict(
            chroms=eval_chroms,
            min_length_match=run_config['eval_length_match'],
            **gm12878_kwargs
        ),
        hic=run_config['hic'],
        progress_bar=run_config['progress_bar'],
        center_context=True,
    )
    
    config = PairEncoderConfig(**model_config)
    model = PairEncoderForClassification(config)

    encoder_path = data_config.data_root + run_config['context_reduce_pretrained']
    load_fn = load_file if encoder_path.endswith('.safetensors') else torch.load
    state_dict = load_fn(encoder_path)
    state_dict = {
        k[len("encoder.context_reduce."):]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.context_reduce.")
    }
    print(model.encoder.context_reduce.load_state_dict(state_dict))
    model = model.cuda()
    
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
