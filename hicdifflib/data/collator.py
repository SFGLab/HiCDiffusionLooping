import torch


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
