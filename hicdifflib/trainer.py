import logging

from torch.utils.data import WeightedRandomSampler
from transformers import Trainer

from hicdifflib.data.dataset import PairedEndsDataset

logger = logging.getLogger(__name__)


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
