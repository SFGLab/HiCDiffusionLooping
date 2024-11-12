import logging
from os import PathLike
from math import floor
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import random
import warnings
import subprocess

from fire import Fire
from dataclasses_json import dataclass_json
from pyranges import PyRanges
from Bio import SeqIO
from tqdm import trange

import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from hicdifflib.utils import read_paired_ends, sequence_to_onehot, sequences_mask, bstr
from hicdifflib.hicdiffusion import HICDIFFUSION_WINDOW_SIZE, HICDIFFUSION_OUTPUT_SIZE


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


@dataclass_json
@dataclass
class DataConfig:
    chroms: list[str] = field(
        default_factory=lambda: [f'chr{i+1}' for i in range(22)]
        # "$(seq -f 'chr%g' 1 22)"
    )
    data_root: str = './data'
    skip_cache: bool = True # some files are too large for keeping 
                            # them in data_root and WANDB_CACHE_DIR
    skip_intermediary: bool = False
    wandb: bool = True
    wandb_project: str = 'HiCDiffusionLooping'
    samtools_exec: str = 'singularity exec samtools.sif samtools'
    delly_exec: str = 'singularity exec delly.sif delly'
    tools_exec: str = 'singularity exec tools.sif'

    def __post_init__(self):
        self._api = wandb.Api(overrides={'project': self.wandb_project})


class JobType:
    DATA_GENERATION: str = 'datagen'
    TRAINING: str = 'training'


class Artifact:
    def __str__(self) -> str:
        ...

    @property
    def path(self) -> Path:
        return Path(str(self))
    
    @property
    def path_no_suffix(self) -> Path:
        return self.path.with_suffix('')


class LocalArtifact(Artifact):
    def __init__(self, path: PathLike) -> None:
        self._path = str(path)
    
    def __str__(self) -> str:
        return self._path



class WandbArtifact(Artifact):
    def __init__(self, name: str, config: DataConfig, run: Run | None) -> None:
        self._artifact: wandb.Artifact = (
            run.use_artifact(name) if run else config._api.artifact(name)
        )
        self._root_dir = self._artifact.download(
            root=config.data_root,
            skip_cache=config.skip_cache,
        )
    
    def __str__(self) -> str:
        return self._artifact.file(self._root_dir)


def _wandb_run(**wandb_kwargs):

    def decorate_run(method: callable) -> callable:

        def decorated_method(self: DataConfig, *args, **kwargs) -> Any:
            if not self.wandb:
                return method(self, None, *args, **kwargs)
            run = wandb.init(
                project=self.wandb_project, 
                name=method.__name__, 
                config=self.to_dict(), 
                **wandb_kwargs
            )
            result = method(self, run, *args, **kwargs)
            run.finish()
            return result
        
        return decorated_method
        
    return decorate_run


def _simple_exec_and_log(cmd_builder: callable) -> callable:

    def decorated_method(self: DataConfig, *args, output_check: Path | None = None, **kwargs):
        name: str = cmd_builder.__name__
        if name.startswith('_exec_'):
            name = name[6:]
        logger = logging.getLogger(name)
        
        args = [str(v) if isinstance(v, (Artifact, Path)) else v for v in args]
        kwargs = {k: (str(v) if isinstance(v, (Artifact, Path)) else v) for k, v in kwargs.items()}
        cmd = cmd_builder(self, *args, **kwargs)

        if output_check and output_check.exists() and not self.skip_intermediary:
            logger.info('Skipping: already %s exists', output_check)
            return
        
        logger.info(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        _log_result(logger, result)
    
    return decorated_method


def _log_text(logg_fn, text):
    for line in text.split('\n'):
        line = line.strip()
        if line:
            logg_fn(line)


def _log_result(logger, result):
    if result.stdout:
        _log_text(logger.info, result.stdout)
    if result.stderr:
        _log_text(logger.warning, result.stderr)
    if result.returncode:
        logger.error('Exited with %d statuscode', result.returncode)
        raise OSError(result.returncode)
    logger.info('Done')


_to_pyrange_columns = {'chr': 'Chromosome', 'start': 'Start', 'end': 'End', 'strand': 'Strand'}


class DataPipeline(DataConfig):
    
    def test_pairs_dataset(self, pet_pairs: list[str | Artifact], sequences: list[str | Artifact]):
        pairs = self._get_artifact(pet_pairs).path
        sequences = [self._get_artifact(fasta).path for fasta in sequences]
        dataset = PairedEndsDataset(pairs=pairs, sequences=sequences, chroms=self.chroms)
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

    @_wandb_run(job_type=JobType.DATA_GENERATION)
    def pet_pairs(
        self, 
        run: Run | None, 
        pairs: list[str | Artifact], 
        peaks: list[str | Artifact], 
        motifs: list[str | Artifact],
        peak_strand_slack: int = 0, 
        min_strand_ratio: float = 0.5,
        anchor_peak_slack: int = 0,
        anchor_peak_min_overlap: int = 500,
    ):
        if self.wandb:
            run.log({
                'peak_strand_slack': peak_strand_slack,
                'min_strand_ratio': min_strand_ratio,
                'anchor_peak_slack': anchor_peak_slack,
                'anchor_peak_min_overlap': anchor_peak_min_overlap,
            })

        pairs = [self._get_artifact(x, run) for x in pairs]
        peaks = [self._get_artifact(x, run) for x in peaks]
        motifs = [self._get_artifact(x, run) for x in motifs]

        pairs_df = (
            pd.concat([
                read_paired_ends(artifact.path, extra_columns=['pet_counts'])
                for artifact in pairs
            ])
            .query('len_full <= @HICDIFFUSION_WINDOW_SIZE')
            .reset_index()
            .rename(columns={'index': 'pair_index'})
        )

        peaks_df = pd.concat([
            pd.read_csv(
                artifact.path, 
                sep='\t', 
                names=['chr', 'start', 'end', 'c1', 'c2', 'c3', 'value', 'c4', 'c5']
            )
            for artifact in peaks
        ])
        peaks_df = peaks_df[['chr', 'start', 'end', 'value']].reset_index()


        motifs_df = pd.concat([
            pd.read_csv(artifact.path, sep='\t', compression='gzip')
            for artifact in motifs
        ])
        motifs_df = motifs_df[motifs_df.sequence_name.isin(self.chroms)]
        motifs_df = (
            motifs_df
            .reset_index(drop=True)
            .rename(columns={'sequence_name': 'chr', 'stop': 'end'})
        )
        motifs_df['matched_len'] = motifs_df['matched_sequence'].map(len)
        motifs_df = motifs_df[['chr', 'start', 'end', 'strand', 'matched_len', 'score']]
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df = self._pet_pairs(
                pairs_df=pairs_df,
                peaks_df=peaks_df,
                motifs_df=motifs_df,
                peak_strand_slack=peak_strand_slack,
                min_strand_ratio=min_strand_ratio,
                anchor_peak_min_overlap=anchor_peak_min_overlap,
                anchor_peak_slack=anchor_peak_slack,
            )
        
        path = Path(self.data_root) / 'pet_pairs.csv'
        df.to_csv(path, index=False)
        self._log_artifact(run, path)


    def _pet_pairs(
        self, 
        pairs_df: pd.DataFrame, 
        peaks_df: pd.DataFrame, 
        motifs_df: pd.DataFrame, 
        peak_strand_slack: int, 
        min_strand_ratio: float,
        anchor_peak_slack: int,
        anchor_peak_min_overlap: int,
    ):
        logger = logging.getLogger('pet_pairs')

        anchors_l_df = pairs_df.rename(
            columns={'chr_l': 'chr', 'start_l': 'start', 'end_l': 'end'}
        )
        anchors_l_df['strand'] = '+'
        anchors_r_df = pairs_df.rename(
            columns={'chr_r': 'chr', 'start_r': 'start', 'end_r': 'end'}
        )
        anchors_r_df['strand'] = '-'
        anchors_df = pd.concat([
            anchors_l_df[['pair_index', 'chr', 'start', 'end', 'strand']], 
            anchors_r_df[['pair_index', 'chr', 'start', 'end', 'strand']]
        ])
        anchors_df = anchors_df.set_index(['pair_index', 'strand'], drop=False)
        anchors_pr = PyRanges(anchors_df.rename(columns=_to_pyrange_columns))
        logger.info('Pairs anchors:')
        _log_text(logger.info, repr(anchors_pr))

        peaks_pr = PyRanges(
            peaks_df.rename(
                columns=dict(index='peak_index', value='peak_value', **_to_pyrange_columns)
            )
        )
        logger.info('Peaks:')
        _log_text(logger.info, repr(peaks_pr))

        motifs_pr = PyRanges(motifs_df.rename(columns=dict(score='motif_score', **_to_pyrange_columns)))
        logger.info('Motifs:')
        _log_text(logger.info, repr(motifs_pr))

        peaks_stranded_pr = peaks_pr.join(
            other=motifs_pr, 
            suffix='_motif', 
            report_overlap=True, 
            slack=peak_strand_slack, 
            apply_strand_suffix=False
        )
        peaks_stranded_pr = peaks_stranded_pr[peaks_stranded_pr.Overlap == peaks_stranded_pr.matched_len-1]
        logger.info('Peaks stranded:')
        _log_text(logger.info, repr(peaks_stranded_pr))

        # calculate confidence of peaks' strand
        df = peaks_stranded_pr.as_df()
        df['left_strand'] = (df['Strand'] == '+').astype(int)
        df['right_strand'] = (df['Strand'] == '-').astype(int)
        dfgb = df.groupby('peak_index').agg(
            count=('Strand', 'count'),
            **{
                '+': ('left_strand', 'mean'),
                '-': ('right_strand', 'mean'),
            }
        )
        confident_strands = (
            dfgb
            .reset_index()
            .melt(
                id_vars=['peak_index'], 
                value_vars=['+', '-'], 
                var_name='Strand', 
                value_name='strand_ratio',
            )
        )
        peaks_stranded_pr = PyRanges(
            confident_strands[confident_strands['strand_ratio'] >= min_strand_ratio]
            .merge(
                on=['peak_index', 'Strand'],
                how='left',
                right=(
                    peaks_stranded_pr
                    .as_df()[['peak_index', 'Chromosome', 'Start', 'End', 'Strand', 'peak_value']]
                    .drop_duplicates(['peak_index', 'Strand'])
                )
            )
        )
        logger.info('Confident peaks stranded:')
        _log_text(logger.info, repr(peaks_stranded_pr))

        l_pr = PyRanges(
            peaks_stranded_pr
            .as_df()[['Chromosome', 'Start', 'End', 'Strand', 'peak_index', 'peak_value']]
            .query('Strand == "+"')
        )
        r_pr = PyRanges(
            peaks_stranded_pr
            .as_df()[['Chromosome', 'Start', 'End', 'Strand', 'peak_index', 'peak_value']]
            .query('Strand == "-"')
        )
        logger.info('Left and right anchors using stranded peaks')
        _log_text(logger.info, repr(l_pr))
        _log_text(logger.info, repr(r_pr))

        anchors_filtered_pr = anchors_pr.join(peaks_stranded_pr, suffix='_peak', report_overlap=True, slack=anchor_peak_slack, strandedness='same', how='right')
        anchors_filtered_pr = anchors_filtered_pr[anchors_filtered_pr.Overlap >= anchor_peak_min_overlap]
        logger.info('Filtered anchors:')
        _log_text(logger.info, repr(anchors_filtered_pr))

        negatives_df = l_pr.join(r_pr, suffix='_r', slack=HICDIFFUSION_WINDOW_SIZE * 2).as_df()
        negatives_df['len_full'] = (negatives_df.End_r - negatives_df.Start)
        negatives_df = negatives_df.query('len_full > 0 & len_full < @HICDIFFUSION_WINDOW_SIZE').reset_index(drop=True)
        logger.info('Generated %d negative pairs', len(negatives_df))
        
        pairs_filtered_df = pairs_df[pairs_df.pair_index.isin(anchors_filtered_pr.pair_index)]
        l_df = (
            anchors_filtered_pr
            .as_df()
            .query('Strand == "+"')[['pair_index', 'peak_index', 'peak_value']]
            .drop_duplicates(['pair_index', 'peak_index'])
        )
        r_df = (
            anchors_filtered_pr
            .as_df()
            .query('Strand == "-"')[['pair_index', 'peak_index', 'peak_value']]
            .drop_duplicates(['pair_index', 'peak_index'])
        )
        positives_df = (
            pairs_filtered_df
            .merge(l_df, on='pair_index')
            .merge(r_df, on='pair_index', suffixes=('_l', '_r'))
        )
        logger.info('Filtered %d positive pairs', len(positives_df))

        filter = pd.Series(zip(positives_df.peak_index_l, positives_df.peak_index_r))
        df_ids = pd.Series(zip(negatives_df.peak_index, negatives_df.peak_index_r))
        negatives_df = negatives_df[~df_ids.isin(filter)].reset_index(drop=True)
        negatives_df['pet_counts'] = 0
        logger.info('After removal of overlapped generated and positive pairs: negative pairs %d', len(negatives_df))
        
        dataset_df = pd.concat([
            positives_df[['chr_l', 'start_l', 'end_l', 'start_r', 'end_r', 'pet_counts', 'peak_value_l', 'peak_value_r']]
            .rename(
                columns={'chr_l': 'chr'}
            ),
            negatives_df[['Chromosome', 'Start', 'End', 'Start_r', 'End_r', 'pet_counts', 'peak_value', 'peak_value_r']]
            .rename(
                columns={'Chromosome': 'chr', 'Start': 'start_l', 'End': 'end_l', 'Start_r': 'start_r', 'End_r': 'end_r', 'peak_value': 'peak_value_l'}
            )
        ])
        return dataset_df
    

    @_wandb_run(job_type=JobType.DATA_GENERATION)
    def fimo(self, run: Run | None, motif: str | Artifact = 'MA0139.2.meme:v0', sequence: str | Artifact = 'GRCh38-reference-genome:v0'):
        motif = self._get_artifact(motif, run)
        sequence = self._get_artifact(sequence, run)
        filename = 'fimo_' + motif.path_no_suffix.name + '_' + sequence.path_no_suffix.name + '.tsv.gz'
        output = Path(self.data_root) / filename
        self._exec_fimo(sequence, motif, output)
        self._log_artifact(run, output)

    @_simple_exec_and_log
    def _exec_fimo(self, fasta: str, motif: str, ouput_gz: str):
        return (
            f'{self.tools_exec} fimo --verbosity 2 --text {motif!r} {fasta!r} | '
            f'{self.tools_exec} gzip > {ouput_gz!r}'
        )

    @_wandb_run(job_type=JobType.DATA_GENERATION)
    def vcf_consensus(self, run: Run | None, variants: str | Artifact, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference, run)
        variants = self._get_artifact(variants, run)
        self.tabix(variants)
        consensus = variants.path.with_suffix('.fa')
        self._exec_vcf_consesus(reference, self.chroms, variants, consensus, output_check=consensus)
        self._log_artifact(run, consensus)
    
    @_simple_exec_and_log
    def _exec_vcf_consesus(self, fasta: str, regions: list[str], vcf: str, output: str):
        regions = ' '.join([repr(r) for r in regions])
        return f'{self.samtools_exec} faidx {fasta!r} {regions} | {self.tools_exec} vcf-consensus {vcf!r} > {output!r}'
    
    def tabix(self, variants: str | Artifact):
        variants = self._get_artifact(variants)
        self._exec_tabix(variants, output_check=variants.path.with_suffix('.gz.tbi'))
    
    @_simple_exec_and_log
    def _exec_tabix(self, vcfgz: str):
        return f'{self.tools_exec} tabix -fp vcf {vcfgz!r}'

    @_wandb_run(job_type=JobType.DATA_GENERATION)
    def delly_call(self, run: Run | None, alignments: str | Artifact, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference, run)
        alignments = self._get_artifact(alignments, run)

        self.index_reference(reference)
        alignments_sorted = self.sort_alignments(alignments)
        self.index_alignments(alignments_sorted)

        delly_vcf = alignments.path.with_suffix('.delly.vcf')
        self._exec_delly_call(reference, alignments_sorted, delly_vcf, output_check=delly_vcf)

        delly_vcfgz = delly_vcf.with_suffix('.vcf.gz')
        self._exec_bgzip(delly_vcf, delly_vcfgz, output_check=delly_vcfgz)

        self._log_artifact(run, delly_vcfgz)

    @_simple_exec_and_log
    def _exec_delly_call(self, fasta: str, bam: str, vcf: str):
        return f'{self.delly_exec} call -g {fasta!r} {bam!r} > {vcf!r}'

    @_simple_exec_and_log
    def _exec_bgzip(self, vcf: str, vcfgz: str):
        return f'{self.tools_exec} bgzip -c {vcf!r} > {vcfgz}'
    

    def index_reference(self, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference)
        self._exec_samtools_faidx(reference, output_check=reference.path.with_suffix('.fa.fai'))

    @_simple_exec_and_log
    def _exec_samtools_faidx(self, fasta: str):
        return f'{self.samtools_exec} faidx {fasta!r}'


    def index_alignments(self, alignments: str | Artifact):
        alignments = self._get_artifact(alignments)
        self._exec_samtools_index(alignments, output_check=alignments.path.with_suffix('.bam.bai'))

    @_simple_exec_and_log
    def _exec_samtools_index(self, bam: str):
        return f'{self.samtools_exec} index {bam!r}'


    def sort_alignments(self, alignments: str | Artifact) -> Path:
        alignments = self._get_artifact(alignments)
        alignments_sorted = alignments.path.with_suffix('.sorted.bam')
        self._exec_samtools_sort(alignments, alignments_sorted, output_check=alignments_sorted)
        return alignments_sorted

    @_simple_exec_and_log
    def _exec_samtools_sort(self, input: str, output: str):
        return f'{self.samtools_exec} sort -o {output!r} {input!r}'


    def _log_artifact(self, run: Run, path: Path):
        if self.wandb:
            result = wandb.Artifact(
                name=path.name,
                type="dataset",
            )
            result.add_file(str(path))
            run.log_artifact(result)

    def _get_artifact(self, artifact: str | Artifact, run: Run | None = None) -> Artifact:
        if isinstance(artifact, Artifact):
            return artifact
        if isinstance(artifact, Path) or ':' not in str(artifact):
            return LocalArtifact(artifact)
        return WandbArtifact(artifact, self, run)


class PairedEndsDataset(Dataset):
    def __init__(
        self,
        pairs: Path,
        sequences: list[Path],
        chroms: list[str],
        mask_size: int = HICDIFFUSION_OUTPUT_SIZE,
        tokenizer: PreTrainedTokenizer | None = None,
        center_context: bool = False
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._pairs_df = (
            pd.read_csv(pairs)
            .query('chr.isin(@chroms)')
        )
        self._pairs_df['label'] = (self._pairs_df['pet_counts'] > 0).astype(int)
        self._sequences = self._load_sequences(sequences, chroms)
        self._tokenizer = tokenizer

        self._logger.info("Validating sequences (%d pairs)", len(self._pairs_df))
        self._valid_sequences = []
        for pair_idx in self._pairs_df.index:
            self._valid_sequences += self._check_pair_sequences(pair_idx)
        self._logger.info("%d valid sequences", len(self._valid_sequences))
        self.mask_size = mask_size
        self.center_context = center_context

    def _load_sequences(self, sequensces: list[Path], chroms: list[str]) -> dict:
        result = defaultdict(list)
        for seq_path in sequensces:
            with open(seq_path) as f:
                records = SeqIO.parse(f, 'fasta')
                for record in records:
                    if record.id not in chroms:
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
            
            # bound possible range of window positions which includes both anchors
            min_context_start = max(row.end_r - HICDIFFUSION_WINDOW_SIZE, 0)
            max_context_end = min(row.start_l + HICDIFFUSION_WINDOW_SIZE, len(seq))
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

    def _context_from(self, max_offset: int, context_offset: float | None = None) -> int:
        if context_offset is not None:
            return floor(max_offset * context_offset) 
        if self.center_context:
            return floor(max_offset * 0.5) 
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
        return data

    def get_item(
        self, 
        i: int, 
        context_offset: float | None = None, 
        tokenize: bool = True, 
        return_tensors: str | None = None
    ) -> dict:
        inputs = self.get_pair_context(i, context_offset)
        
        if self._tokenizer and tokenize:
            left = self._tokenizer(text=str(inputs['anchor_l']), return_tensors=return_tensors)
            right = self._tokenizer(text=str(inputs['anchor_r']), return_tensors=return_tensors)
            inputs['left_input_ids'] = left['input_ids']
            inputs['right_input_ids'] = right['input_ids']
            
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
        return inputs

    def __getitem__(self, i: int) -> dict:
        try:
            inputs = self.get_item(i)
        except Exception as e:
            print(i)
            raise e
        return inputs

        

if __name__ == '__main__':
    Fire(DataPipeline)
