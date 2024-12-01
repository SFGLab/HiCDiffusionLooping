import logging

import torch
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer

from hicdifflib.data import PairedEndsDataset


logger = logging.getLogger(__name__)

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


class BalancedTrainer(Trainer):
    def _get_train_sampler(self):
        self.train_dataset: PairedEndsDataset
        labels = self.train_dataset.get_labels()
        return WeightedRandomSampler(
            weights=make_weights_for_balanced_classes(labels, 2),
            num_samples=int(sum(labels) * 2),
            replacement=False
        )
