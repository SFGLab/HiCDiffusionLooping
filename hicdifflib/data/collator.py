import torch


class PairedEndsCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        left_sequences = []
        right_sequences = []
        labels = []
        masks = []
        contexts = []
        hics = []
        for sample in features:
            left_sequences.append(sample['left_sequence'])
            right_sequences.append(sample['right_sequence'])
            labels.append([sample['label']])
            masks.append(sample['context_mask'])
            contexts.append(sample['context_sequence'])
            if sample['hic'] is not None:
                hics.append(sample['hic'])

        # left = self.tokenizer.pad({'input_ids': left_sequences}, return_tensors='pt')
        # right = self.tokenizer.pad({'input_ids': right_sequences}, return_tensors='pt')
        left = self.tokenizer(text=left_sequences, padding=True, return_tensors='pt')
        right = self.tokenizer(text=right_sequences, padding=True, return_tensors='pt')
            
        batch = {
            'left_sequence': left['input_ids'],
            'left_attention_mask': left['attention_mask'],
            'right_sequence': right['input_ids'],
            'right_attention_mask': right['attention_mask'],
            'labels': torch.tensor(labels),
            'context_sequence': torch.cat(contexts, dim=0),
            'context_mask': torch.cat(masks, dim=0),
            'hic': torch.cat(hics, dim=0) if hics else None,
        }
        return batch
