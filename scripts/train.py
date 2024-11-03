#!/usr/bin/env python
# coding: utf-8

# In[1]:

from hicdifflib.data import PairedEndsDataset
from hicdifflib.dnabert import PairEncoderConfig, PairEncoderForClassification


# In[2]:


from pathlib import Path

import wandb

import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel, AutoTokenizer

api = wandb.Api(overrides={'project': 'HiCDiffusionLooping'})
hicdiffustion_artifact = api.artifact('hicdiffusion_encoder_decoder:v0')
hicdiffustion_artifact = api.artifact('hicdiffusion_encoder_decoder:v0')
# hicdiffustion_artifact.download('.b')


# In[3]:


config = PairEncoderConfig(hicdiffusion_checkpoint=hicdiffustion_artifact.file('.'))
config


# In[4]:


# saved chr14 for testing and chr15 for validation.
test_chroms = ['chr14']
eval_chroms = ['chr15']
train_chroms = [f'chr{i}' for i in range(1, 23) if f'chr{i}' not in (test_chroms+eval_chroms)]


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("m10an/DNABERT-S", trust_remote_code=True)
common = dict(
    pairs=api.artifact('pet_pairs.csv:latest').file('.'),
    sequences=[
        # Path(api.artifact('4DNFI1GNQM8L.delly.vcf.fa:latest').file('.')),
        Path(api.artifact('GRCh38-reference-genome:v0').file('.'))
    ],
    tokenizer=tokenizer
)
train_dataset = PairedEndsDataset(
    chroms=train_chroms,
    **common,
)
eval_dataset = PairedEndsDataset(
    chroms=eval_chroms,
    **common
)
# 37118, 1568
len(train_dataset), len(eval_dataset)


# In[ ]:


model = PairEncoderForClassification(config).cuda()


# In[ ]:


# with torch.inference_mode():
#     result = model(
#         left_ids=inputs['left_ids'].cuda(),
#         right_ids=inputs['right_ids'].cuda(),
#         context_sequence=inputs['context_sequence'].to(torch.float32).cuda(),
#         context_mask=inputs['context_mask'].to(torch.float32).cuda(),
#         labels=torch.tensor([[label]]).to(torch.float32).cuda()
#     )


# In[ ]:


from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir="./test", 
    evaluation_strategy='epoch',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
) 
training_args


# In[ ]:


from torch.utils.data import WeightedRandomSampler

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
        return WeightedRandomSampler(
            weights=make_weights_for_balanced_classes(self.train_dataset._pairs_df['label'], 2),
            num_samples=int(self.train_dataset._pairs_df['label'].sum() * 2),
            replacement=False
        )


# In[ ]:


train_dataset[0]


# In[ ]:


# from transformers.data.data_collator import pad_without_fast_tokenizer_warning

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
            left_input_ids += sample['left_input_ids'].tolist()
            right_input_ids += sample['right_input_ids'].tolist()
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


# In[ ]:


trainer = CustomTrainer(
    model=model,
    data_collator=PairedEndsCollatorWithPadding(tokenizer),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics
)


# In[ ]:


res = trainer.train()


# In[ ]:




