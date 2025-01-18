import os
import sys
import logging
import datetime as dt
from pathlib import Path

import wandb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Subset
from transformers import AutoTokenizer
from transformers import TrainingArguments

from hicdifflib.data.base import DataConfig, WandbArtifact
from hicdifflib.data.collator import PairedEndsCollatorWithPadding
from hicdifflib.data.dataset import PairedEndsDataset
from hicdifflib.v2 import PairEncoderConfig2 as PairEncoderConfig, PairEncoderForClassification
from hicdifflib.metrics import compute_metrics
from hicdifflib.trainer import BalancedTrainer


logger = logging.getLogger(__name__)

DEBUG=False
MODEL_NAME=sys.argv[1]
RUN_NAME=MODEL_NAME
RUN_CHECKPOINT=int(sys.argv[2])

if DEBUG:
    RUN_NAME += '_debug'

run_config = dict(
    save_code=True, 
    job_type='predict' if not DEBUG else 'debug',
    project='HiCDiffusionLooping',
    name=f"predict_{RUN_NAME}",
    group=RUN_NAME,
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(run):
    test_chroms = ['chr14']
    eval_chroms = ['chr15']
    train_chroms = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in (test_chroms+eval_chroms)]
    
    tokenizer = AutoTokenizer.from_pretrained("m10an/DNABERT-S", trust_remote_code=True)
    data_config = DataConfig(data_root='/mnt/evafs/scratch/shared/imialeshka/hicdata/')

    dataset_kwargs = dict(
        pairs=WandbArtifact('gm12878_pairs.csv:v4', data_config, run).path,
        # pairs=WandbArtifact('pet_pairs.csv:v1', data_config, run).path,
        sequences=[WandbArtifact('GRCh38-reference-genome:v0', data_config, run).path],
        tokenizer=tokenizer,
        progress_bar=False,
        positive_min_pet_counts=3,
        negative_max_pet_counts=2,
        max_anchor_length=510,
        # hic=WandbArtifact('gm12878_4DNFIUEG39YZ:v0', data_config).path,
    )
    
    for key, value in os.environ.items():
        if key.startswith('SLURM_'):
            run.config[key] = value
    
    for key, value in dataset_kwargs.items():
        if isinstance(value, (int, float, str, Path)):
            run.config[key] = value

    pred_filename = f"preds_{dt.datetime.now().isoformat(sep='_', timespec='seconds')}.csv"
    run.config.update({
        'model_name': MODEL_NAME, 'checkpoint': RUN_CHECKPOINT, 'predictions_file': pred_filename
    })

    model_path = f'/mnt/evafs/scratch/shared/imialeshka/hicdata/{MODEL_NAME}/checkpoint-{RUN_CHECKPOINT}/'
    preds_path = model_path + pred_filename
    model = PairEncoderForClassification.from_pretrained(model_path).cuda()
    run.log({
        'n_parameters': count_parameters(model),
        'n_parameters_trainable': count_trainable_parameters(model)
    })
    
    eval_dataset = PairedEndsDataset(
        chroms=eval_chroms+test_chroms,
        center_context=True,
        **dataset_kwargs
    )
    if DEBUG:
        eval_dataset._valid_sequences = eval_dataset._valid_sequences[:20]
    
    test_samples = pd.DataFrame(eval_dataset._valid_sequences)
    test_samples = eval_dataset._pairs_df.merge(
        right=test_samples,
        right_on='pair_idx',
        left_index=True,
        how='right'
    )
    
    dl = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=8,
        num_workers=8,
        shuffle=False,
        collate_fn=PairedEndsCollatorWithPadding(tokenizer)
    )
    test_logits = []
    for batch in tqdm(dl):
        with torch.inference_mode():
            loss, logits = model(**{k: (v.cuda() if v is not None else v) for k,v in batch.items()})
        test_logits.append(logits.cpu().numpy())
    test_samples['logits'] = np.concatenate(test_logits)
    test_samples.to_csv(preds_path, index=False)

    valid = test_samples[test_samples.chr.isin(eval_chroms)]
    valid_metrics = compute_metrics((
        np.expand_dims(valid.logits.values, 1),
        np.expand_dims(valid.label.values, 1)
    ))
    run.log({f'valid/{k}': v for k,v in valid_metrics.items()})
    run.log({f'valid_gm12878/{k}': v for k,v in valid_metrics.items()})
    
    test = test_samples[test_samples.chr.isin(test_chroms)]
    test_metrics = compute_metrics(
        (
            np.expand_dims(test.logits.values, 1),
            np.expand_dims(test.label.values, 1)
        ),
        thr=valid_metrics['threshold']
    )
    run.log({f'test/{k}': v for k,v in test_metrics.items()})
    run.log({f'test_gm12878/{k}': v for k,v in test_metrics.items()})
    


if __name__ == '__main__':
    with wandb.init(**run_config) as run:
        main(run)
