import logging
from pathlib import Path

import wandb
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers.models.bert.configuration_bert import BertConfig
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from hicdifflib.data import PairedEndsDataset, DataConfig, WandbArtifact
from hicdifflib.dnabert import PairEncoderConfig, PairEncoderForClassification
from hicdifflib.hicdiffusion import HICDIFFUSION_WINDOW_SIZE

logger = logging.getLogger(__name__)
run_config = dict(
    save_code=True, 
    project='HiCDiffusionLooping',
    name="train_dnabert",
    tags=["baseline"],
    group="dnabert",
)

def main(run):
    test_chroms = ['chr14']
    eval_chroms = ['chr15']
    # train_chroms = ['chr1']
    train_chroms = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in (test_chroms+eval_chroms)]
    training_args = TrainingArguments(
        output_dir="/mnt/evafs/scratch/shared/imialeshka/hicdata/test2",
        num_train_epochs=10,
        evaluation_strategy='steps',
        eval_steps=10000,
        eval_delay=0,
        save_strategy='steps',
        save_steps=10000,
        logging_strategy='steps',
        logging_steps=10,
        auto_find_batch_size=False,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        run_name='dnabert',
        disable_tqdm=False,
        dataloader_num_workers=8,
        metric_for_best_model='f1',
        greater_is_better=True,
    )
        
    data_config = DataConfig(data_root='/mnt/evafs/scratch/shared/imialeshka/hicdata/')
    config = PairEncoderConfig(
        anchor_encoder_shared=True,
        hicdiffusion_frozen=False,
        hicdiffusion_checkpoint=str(WandbArtifact('hicdiffusion_encoder_decoder:v0', data_config))
    )
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
        # max_anchor_length=510,
    )
    
    
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


class PairedEndsCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: list[dict]) -> dict:
        left_input_ids = []
        right_input_ids = []
        labels = []
        masks = []
        contexts = []
        for sample in features:
            left_input_ids.append(sample['left_input_ids'])
            right_input_ids.append(sample['right_input_ids'])
            labels.append([sample['label']])
            masks.append(sample['context_mask'])
            contexts.append(sample['context_sequence'])
            
        left = self.tokenizer.pad({'input_ids': left_input_ids}, return_tensors='pt')
        right = self.tokenizer.pad({'input_ids': right_input_ids}, return_tensors='pt')
        batch = {
            'left_input_ids': left['input_ids'],
            'left_attention_mask': left['attention_mask'],
            'right_input_ids': right['input_ids'],
            'right_attention_mask': right['attention_mask'],
            'labels': torch.tensor(labels),
            'context_sequence': torch.cat(contexts, dim=0),
            'context_mask': torch.cat(masks, dim=0),
        }
        return batch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(logits[:, 0]))
    predictions = (probabilities >= 0.5).astype(int)

    try:
        precision = precision_score(labels, predictions)
    except Exception as e:
        precision = np.nan
        logger.warning(str(e))
    
    try:
        recall = recall_score(labels, predictions)
    except Exception as e:
        recall = np.nan
        logger.warning(str(e))
    
    try:
        f1 = f1_score(labels, predictions)
    except Exception as e:
        f1 = np.nan
        logger.warning(str(e))
    
    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except Exception as e:
        roc_auc = np.nan
        logger.warning(str(e))
    
    try:
        pr_auc = average_precision_score(labels, probabilities)
    except Exception as e:
        pr_auc = np.nan
        logger.warning(str(e))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": pr_auc
    }


def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    for label in labels:                                                         
        count[label] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, label in enumerate(labels):                                          
        weight[idx] = weight_per_class[label]                                  
    return weight

class CustomTrainer(Trainer):
    def _get_train_sampler(self):
        self.train_dataset: PairedEndsDataset
        labels = [int(self.train_dataset._pairs_df.loc[x['pair_idx'], 'label']) for x in self.train_dataset._valid_sequences]
        return WeightedRandomSampler(
            weights=make_weights_for_balanced_classes(labels, 2),
            num_samples=int(sum(labels) * 2),
            replacement=False
        )

if __name__ == '__main__':
    with wandb.init(**run_config) as run:
        main(run)
