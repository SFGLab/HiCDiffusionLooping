import random
import logging
from math import floor
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from Bio import SeqIO
from cooler import Cooler
from torch.utils.data import Dataset, ConcatDataset
from fire import Fire
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizer, AutoTokenizer
from skimage.transform import resize

from hicdifflib.hicdiffusion import HICDIFFUSION_OUTPUT_SIZE, HICDIFFUSION_WINDOW_SIZE
from hicdifflib.data.base import Artifact
from hicdifflib.data.pipeline import DataPipeline
from hicdifflib.utils import bstr, sequence_to_onehot, sequences_mask

HIC_SHAPE = (HICDIFFUSION_OUTPUT_SIZE, HICDIFFUSION_OUTPUT_SIZE)


class ConcatPairedEndsDataset(ConcatDataset):
    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)
        self._datasets = datasets
        self._labels = []
        for ds in self._datasets:
            self.labels.append(ds.get_labels())

    def get_labels(self) -> list[int]:
        return self._labels


def load_sequences(seq_path: Path, chroms: list[str]) -> dict:
    result = {}
    with open(seq_path) as f:
        records = SeqIO.parse(f, 'fasta')
        for record in records:
            if record.id not in chroms:
                continue
            result[record.id] = record.seq
    return result


class FilteredPairedEndsDataset(Dataset):
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        sequence: Path | dict,
        hic: Path | None = None,
        hic_resolution: int = 10_000,
        mask_size: int = HICDIFFUSION_OUTPUT_SIZE,
        tokenizer: PreTrainedTokenizer | None = None,
        center_context: bool = False,
        center_position: float = 0.5,
        progress_bar: bool = True,
    ) -> None:
        self.mask_size = mask_size
        self.center_context = center_context
        self.center_position = center_position
        self._tokenizer = tokenizer
        self._cooler = None if hic is None else Cooler(f"{hic}::/resolutions/{hic_resolution}")
        self._logger = logging.getLogger(self.__class__.__name__)
        self._sequence = sequence if isinstance(sequence, dict) else load_sequences(sequence, chroms)
        self._pairs_df = self.pairs_df

    def __len__(self):
        return len(self._pairs_df)

    def __getitem__(self, i: int) -> dict:
        try:
            inputs = self.get_item(i)
        except Exception as e:
            print(i)
            raise e
        return inputs

    def get_item(
        self,
        i: int,
        context_offset: float | None = None,
        tokenize: bool = True,
        return_tensors: str | None = None
    ) -> dict:
        inputs = self.get_pair_context(i, context_offset)
        inputs['left_sequence'] = str(inputs['anchor_l'])
        inputs['right_sequence'] = str(inputs['anchor_r'])
        inputs['context_sequence'] = torch.unsqueeze(
            input=sequence_to_onehot(str(inputs['context'])),
            dim=0
        )
        inputs['context_mask'] = sequences_mask(
            n=len(inputs['context']),
            start_l=inputs['anchor_slice_l'].start,
            end_l=inputs['anchor_slice_l'].stop,
            start_r=inputs['anchor_slice_r'].start,
            end_r=inputs['anchor_slice_r'].stop,
            size=self.mask_size
        )

        if inputs['hic'] is not None:
            inputs['hic'] = torch.unsqueeze(torch.unsqueeze(torch.tensor(inputs['hic'], dtype=torch.float32), 0), 0)
        return inputs

    def get_labels(self) -> list[int]:
        return self._pairs_df.label.tolist()

    def _context_from(self, max_offset: int, context_offset: float | None = None) -> int:
        if context_offset is not None:
            return floor(max_offset * context_offset)
        if self.center_context:
            return floor(max_offset * self.center_position)
        return random.randint(0, max_offset)

    def get_pair_context(self, i: int, context_offset: float | None = None) -> dict:
        row = self._pairs_df.loc[i]

        context_from = self._context_from(
            max_offset=row['max_context_start'] - row['min_context_start'],
            context_offset=context_offset
        )
        context_start = row['min_context_start'] + context_from
        context_end = context_start + HICDIFFUSION_WINDOW_SIZE
        seq = self._sequence[row['chr']]

        data = dict(
            sample_idx=i,
            chr=row.chr,
            context_slice=slice(context_start, context_end),
            anchor_slice_l=slice(row.start_l - context_start, row.end_l - context_start),
            anchor_slice_r=slice(row.start_r - context_start, row.end_r - context_start),
            label=float(row.label),
        )
        data['context'] = seq[data['context_slice']]
        data['anchor_l'] = data['context'][data['anchor_slice_l']]
        data['anchor_r'] = data['context'][data['anchor_slice_r']]

        data['hic'] = None
        if self._cooler is not None:
            data['hic'] = resize(self.read_hic_matrix(data['chr'], context_start, context_end), HIC_SHAPE)
        return data

    def read_hic_matrix(self, chr, start, end):
        m = self._cooler.matrix(field="count", balance=None).fetch(f"{chr}:{start}-{end}")
        return np.log(m+1)


class PairedEndsDataset(Dataset):
    def __init__(
        self,
        pairs: Path,
        sequences: list[Path],
        chroms: list[str],
        hic: Path | None = None,
        hic_resolution: int = 10_000,
        mask_size: int = HICDIFFUSION_OUTPUT_SIZE,
        tokenizer: PreTrainedTokenizer | None = None,
        center_context: bool = False,
        center_position: float = 0.5,
        min_length_match: float = 0.95,
        max_anchor_length: int = 512,
        progress_bar: bool = True,
        positive_min_pet_counts: int = 1,
        negative_max_pet_counts: int = 0,
        # cached data
        pairs_df: pd.DataFrame | None = None,
        valid_sequences: list | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.mask_size = mask_size
        self.center_context = center_context
        self.center_position = center_position
        self.min_length_match = min_length_match
        self.max_anchor_length = max_anchor_length
        self.progress_bar = progress_bar
        self._cooler = None if hic is None else Cooler(f"{hic}::/resolutions/{hic_resolution}")
        self._logger = logging.getLogger(self.__class__.__name__)
        self._sequences = self._load_sequences(sequences, chroms)


        if positive_min_pet_counts <= negative_max_pet_counts:
            raise ValueError(
                f'Labels therhold are intesecting: '
                f'{positive_min_pet_counts} <= {negative_max_pet_counts:}'
            )

        if pairs_df is not None:
            self._pairs_df = pairs_df
            self._valid_sequences = valid_sequences
            if not valid_sequences:
                self._valid_sequences = []
                for pair_idx in tqdm(self._pairs_df.index, disable=not self.progress_bar, ):
                    self._valid_sequences += self._check_pair_sequences(pair_idx)
                self._logger.info("%d valid sequences", len(self._valid_sequences))
            return
        
        self._pairs_df = (
            pd.read_csv(pairs)
            .query('chr.isin(@chroms)')
        )
        self._pairs_df['label'] = (self._pairs_df['pet_counts'] >= positive_min_pet_counts).astype(int)
        margin_labels =(
            (self._pairs_df['pet_counts'] < positive_min_pet_counts) &
            (self._pairs_df['pet_counts'] > negative_max_pet_counts)
        )
        self._logger.info("thesholded %d margin pairs", sum(margin_labels))
        self._pairs_df = self._pairs_df.loc[~margin_labels, :]
        self._logger.info("positives: %s%%", 100 * sum(self._pairs_df['label']) / len(self._pairs_df))

        self._logger.info("Validating sequences (%d pairs)", len(self._pairs_df))
        self._valid_sequences = []
        for pair_idx in tqdm(self._pairs_df.index, disable=not self.progress_bar, ):
            self._valid_sequences += self._check_pair_sequences(pair_idx)
        self._logger.info("%d valid sequences", len(self._valid_sequences))

    def read_hic_matrix(self, chr, start, end):
        m = self._cooler.matrix(field="count", balance=None).fetch(f"{chr}:{start}-{end}")
        return np.log(m+1)

    def _load_sequences(self, sequensces: list[Path], chroms: list[str]) -> dict:
        result = defaultdict(list)
        def _reference_match(record):
            ref_len = len(result[record.id][0])
            rec_len = len(record)
            return min(ref_len / rec_len, rec_len / ref_len)

        for seq_idx, seq_path in enumerate(sequensces):
            with open(seq_path) as f:
                records = SeqIO.parse(f, 'fasta')
                for record in records:
                    if record.id not in chroms:
                        continue
                    if seq_idx > 0 and (match := _reference_match(record)) < self.min_length_match:
                        self._logger.info(
                            "Skipping '%s' from '%s' (%s), (%d%% match with reference)",
                            record.id, seq_path.name, bstr(len(record.seq)), round(100*match)
                        )
                        continue
                    result[record.id].append(record.seq)
                    self._logger.info("Loaded '%s' from '%s' (%s)", record.id, seq_path.name, bstr(len(record.seq)))
        return result

    def __len__(self):
        return len(self._valid_sequences)


    def _check_pair_sequences(self, pair_idx: int) -> list[dict]:
        row = self._pairs_df.loc[pair_idx]
        result = []
        for seq_idx, seq in enumerate(self._sequences[row.chr]):
            # check if any index is higher than a sequence
            if any(x >= len(seq) for x in [row.start_l, row.end_l - 1, row.start_r, row.end_r - 1]):
                continue

            if self._tokenizer and any(
                len(self._tokenizer.tokenize(str(seq[anchor]))) > self.max_anchor_length
                for anchor in [slice(row.start_l, row.end_l), slice(row.start_r, row.end_r)]
            ):
                continue

            # bound possible range of window positions which includes both anchors
            min_context_start = max(row.end_r - HICDIFFUSION_WINDOW_SIZE, 0)
            max_context_end = min(
                row.start_l + HICDIFFUSION_WINDOW_SIZE, 
                len(seq), 
                *([] if self._cooler is None else [self._cooler.chromsizes[row.chr]])
            )
            if (
                min_context_start > row.start_l or
                max_context_end < row.end_r or
                max_context_end - min_context_start < HICDIFFUSION_WINDOW_SIZE
            ):
                continue
            result.append({
                'pair_idx': pair_idx,
                'seq_idx': seq_idx,
                'min_context_start': min_context_start,
                'max_context_start': max_context_end - HICDIFFUSION_WINDOW_SIZE
            })
        return result

    def get_labels(self) -> list[int]:
        return [
            int(self._pairs_df.loc[x['pair_idx'], 'label'])
            for x in self._valid_sequences
        ]

    def _context_from(self, max_offset: int, context_offset: float | None = None) -> int:
        if context_offset is not None:
            return floor(max_offset * context_offset)
        if self.center_context:
            return floor(max_offset * self.center_position)
        return random.randint(0, max_offset)

    def get_pair_context(self, i: int, context_offset: float | None = None) -> dict:
        valid_seq = self._valid_sequences[i]
        row = self._pairs_df.loc[valid_seq['pair_idx']]

        context_from = self._context_from(
            max_offset=valid_seq['max_context_start'] - valid_seq['min_context_start'],
            context_offset=context_offset
        )
        context_start = valid_seq['min_context_start'] + context_from
        context_end = context_start + HICDIFFUSION_WINDOW_SIZE
        seq = self._sequences[row.chr][valid_seq['seq_idx']]

        data = dict(
            sample_idx=i,
            pair_idx=valid_seq['pair_idx'],
            seq_idx=valid_seq['seq_idx'],
            chr=row.chr,
            context_slice=slice(context_start, context_end),
            anchor_slice_l=slice(row.start_l - context_start, row.end_l - context_start),
            anchor_slice_r=slice(row.start_r - context_start, row.end_r - context_start),
            label=float(row.label),
        )
        data['context'] = seq[data['context_slice']]
        data['anchor_l'] = data['context'][data['anchor_slice_l']]
        data['anchor_r'] = data['context'][data['anchor_slice_r']]
        
        data['hic'] = None
        if self._cooler is not None:
            data['hic'] = resize(self.read_hic_matrix(data['chr'], context_start, context_end), HIC_SHAPE)
        return data

    def get_item(
        self,
        i: int,
        context_offset: float | None = None,
        tokenize: bool = True,
        return_tensors: str | None = None
    ) -> dict:
        inputs = self.get_pair_context(i, context_offset)

        # left = str(inputs['anchor_l'])
        # right = str(inputs['anchor_r'])
        # if self._tokenizer and tokenize:
        #     left = self._tokenizer(text=left, return_tensors=return_tensors)['input_ids']
        #     right = self._tokenizer(text=right, return_tensors=return_tensors)['input_ids']
        inputs['left_sequence'] = str(inputs['anchor_l'])
        inputs['right_sequence'] = str(inputs['anchor_r'])

        inputs['context_sequence'] = torch.unsqueeze(
            input=sequence_to_onehot(str(inputs['context'])),
            dim=0
        )
        inputs['context_mask'] = sequences_mask(
            n=len(inputs['context']),
            start_l=inputs['anchor_slice_l'].start,
            end_l=inputs['anchor_slice_l'].stop,
            start_r=inputs['anchor_slice_r'].start,
            end_r=inputs['anchor_slice_r'].stop,
            size=self.mask_size
        )

        if inputs['hic'] is not None:
            inputs['hic'] = torch.unsqueeze(torch.unsqueeze(torch.tensor(inputs['hic'], dtype=torch.float32), 0), 0)
        return inputs

    def __getitem__(self, i: int) -> dict:
        try:
            inputs = self.get_item(i)
        except Exception as e:
            print(i)
            raise e
        return inputs


class DatasetTester(DataPipeline):
    def test_pairs_dataset(self, pet_pairs: list[str | Artifact], sequences: list[str | Artifact]):
        pairs = self._get_artifact(pet_pairs).path
        sequences = [self._get_artifact(fasta).path for fasta in sequences]
        dataset = PairedEndsDataset(
            pairs=pairs, sequences=sequences, chroms=self.chroms,
            tokenizer=AutoTokenizer.from_pretrained("m10an/DNABERT-S", trust_remote_code=True)
        )
        for i in trange(len(dataset)):
            for context_offset in (0, 0.5, 1):
                data = dataset.get_pair_context(i, context_offset)
                row = dataset._pairs_df.loc[data['pair_idx']]
                seq = dataset._sequences[row.chr][data['seq_idx']]

                anchor_test = seq[row.start_l: row.end_l]
                if data['anchor_l'] != anchor_test:
                    data['anchor_l_len'] = len(data['anchor_l'])
                    data['anchor_l_test'] = anchor_test
                    data['anchor_l_test_len'] = len(anchor_test)
                    raise AssertionError(context_offset, data)

                anchor_test = seq[row.start_r: row.end_r]
                if data['anchor_r'] != anchor_test:
                    data['anchor_r_len'] = len(data['anchor_r'])
                    data['anchor_r_test'] = anchor_test
                    data['anchor_r_test_len'] = len(anchor_test)
                    raise AssertionError(context_offset, data)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    Fire(DatasetTester)
