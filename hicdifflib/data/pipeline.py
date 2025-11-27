import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import wandb
from Bio import SeqIO
from fire import Fire
from transformers import AutoTokenizer
from dataclasses_json import dataclass_json
from wandb.sdk.wandb_run import Run

from hicdifflib.data import CHROM_SETS
from hicdifflib.data.base import Artifact, LocalArtifact, WandbArtifact, JobType
from hicdifflib.data.base import wandb_run, simple_exec_and_log
from hicdifflib.data.petutils import generate_pet_pairs, check_pair_sequences
from hicdifflib.hicdiffusion import HICDIFFUSION_WINDOW_SIZE
from hicdifflib.utils import read_paired_ends, bstr


logger = logging.getLogger(__name__)

@dataclass_json
@dataclass
class DataConfig:
    chroms: list[str] = field(
        default_factory=lambda: [f'chr{i+1}' for i in range(22)]
        # "$(seq -f 'chr%g' 1 22)"
    )
    chroms_set: str | None = None
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
        if self.chroms_set is not None:
            self.chroms = CHROM_SETS[self.chroms_set]

class DataPipeline(DataConfig):
    
    @wandb_run(job_type=JobType.DATA_GENERATION)
    def filter_pairs(
        self,
        run: Run | None,
        pet_pairs: str | Artifact,
        sequence: str | Artifact,
        reference: str | Artifact = 'GRCh38-reference-genome:v0',
        tokenizer_name: str = 'm10an/DNABERT-S',
        min_chrom_length_match: float = 0.95,
        max_anchor_tokens: int = 510,
        name: str = 'filtered',
    ):
        if self.wandb:
            run.config.update({
                'min_chrom_length_match': min_chrom_length_match,
                'max_anchor_tokens': max_anchor_tokens,
                'tokenizer_name': tokenizer_name,
            })
            run.name = 'filter_pairs_' + name
        pet_pairs = self._get_artifact(pet_pairs, run)
        sequence = self._get_artifact(sequence, run)
        reference = self._get_artifact(reference, run)
        
        reference_seq = {}
        with open(reference.path) as f:
            records = SeqIO.parse(f, 'fasta')
            for record in records:
                if record.id not in self.chroms:
                    continue
                reference_seq[record.id] = record.seq
                logger.info("Loaded '%s' from reference (%s)", record.id, bstr(len(record.seq)))
        
        if reference == sequence:
            seq = reference_seq
        else:
            seq = {}
            with open(sequence.path) as f:
                records = SeqIO.parse(f, 'fasta')
                for record in records:
                    if record.id not in self.chroms:
                        continue
                    ref_len = len(reference_seq[record.id])
                    rec_len = len(record)
                    ratio = min(ref_len / rec_len, rec_len / ref_len)
                    if ratio < min_chrom_length_match:
                        logger.info(
                            "Skipping '%s' (%s), (%d%% match with reference)",
                            record.id, bstr(len(record.seq)), round(100*ratio)
                        )
                        continue
                    seq[record.id] = record.seq
                    logger.info("Loaded '%s' (%s)", record.id, bstr(len(record.seq)))

        df = check_pair_sequences(
            pet_pairs=pd.read_csv(pet_pairs.path),
            chrom_sequences=seq,
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True),
            max_anchor_tokens=max_anchor_tokens,
            progress_bar=False,
        )
        
        path = Path(self.data_root) / f'{name}_pairs.csv'
        df.to_csv(path, index=False)
        self._log_artifact(run, path)
    
    @wandb_run(job_type=JobType.DATA_GENERATION)
    def pet_pairs(
        self,
        run: Run | None,
        pairs: list[str | Artifact],
        peaks: list[str | Artifact],
        motifs: list[str | Artifact],
        peak_score_idx: int = 6,
        peak_strand_slack: int = 0,
        min_strand_ratio: float = 0.5,
        anchor_peak_slack: int = 0,
        anchor_peak_min_overlap: int = 500,
        knn_algorithm: str = 'ball_tree', 
        knn_radius: float = 10_000, 
        knn_metric: str = 'euclidean',
        knn_distance_to_drop: float | None = None,
        name: str = 'pet',
    ):
        if self.wandb:
            run.config.update({
                'peak_strand_slack': peak_strand_slack,
                'min_strand_ratio': min_strand_ratio,
                'anchor_peak_slack': anchor_peak_slack,
                'anchor_peak_min_overlap': anchor_peak_min_overlap,
            })
            if knn_distance_to_drop is not None:
                run.config.update({
                    'knn_distance_to_drop': knn_distance_to_drop,
                    'knn_metric': knn_metric,
                    'knn_radius': knn_radius,
                    'knn_algorithm': knn_algorithm,
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
                header=None,
                # names=['chr', 'start', 'end', 'c1', 'c2', 'c3', 'value', 'c4', 'c5']
            )
            for artifact in peaks
        ])
        peaks_df = peaks_df.loc[:, [0, 1, 2, peak_score_idx]]
        peaks_df.columns = ['chr', 'start', 'end', 'value']
        peaks_df = peaks_df.reset_index()

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
            df = generate_pet_pairs(
                pairs_df=pairs_df,
                peaks_df=peaks_df,
                motifs_df=motifs_df,
                peak_strand_slack=peak_strand_slack,
                min_strand_ratio=min_strand_ratio,
                anchor_peak_min_overlap=anchor_peak_min_overlap,
                anchor_peak_slack=anchor_peak_slack,
                negative_pair_range=HICDIFFUSION_WINDOW_SIZE,
                knn_distance_to_drop=knn_distance_to_drop,
                knn_metric=knn_metric,
                knn_radius=knn_radius,
                knn_algorithm=knn_algorithm,
            )

        path = Path(self.data_root) / f'{name}_pairs.csv'
        df.to_csv(path, index=False)
        self._log_artifact(run, path)

    
    ##################### fimo #####################################################################

    @wandb_run(job_type=JobType.DATA_GENERATION)
    def fimo(self, run: Run | None, motif: str | Artifact = 'MA0139.2.meme:v0', sequence: str | Artifact = 'GRCh38-reference-genome:v0'):
        motif = self._get_artifact(motif, run)
        sequence = self._get_artifact(sequence, run)
        filename = 'fimo_' + motif.path_no_suffix.name + '_' + sequence.path_no_suffix.name + '.tsv.gz'
        subdir = sequence.path.relative_to(self.data_root).parent
        output = Path(self.data_root) / subdir / filename
        self._exec_fimo(sequence, motif, output)
        self._log_artifact(run, output)

    @simple_exec_and_log
    def _exec_fimo(self, fasta: str, motif: str, ouput_gz: str):
        return (
            f'{self.tools_exec} fimo --verbosity 2 --text {motif!r} {fasta!r} | '
            f'{self.tools_exec} gzip > {ouput_gz!r}'
        )


    ##################### vcf_consensus ############################################################
    
    @wandb_run(job_type=JobType.DATA_GENERATION)
    def vcf_consensus(self, run: Run | None, variants: str | Artifact, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference, run)
        variants = self._get_artifact(variants, run)
        self.tabix(variants)
        consensus = variants.path.with_suffix('.fa')
        self._exec_vcf_consesus(reference, self.chroms, variants, consensus, output_check=consensus)
        self._log_artifact(run, consensus)

    @simple_exec_and_log
    def _exec_vcf_consesus(self, fasta: str, regions: list[str], vcf: str, output: str):
        regions = ' '.join([repr(r) for r in regions])
        return f'{self.samtools_exec} faidx {fasta!r} {regions} | {self.tools_exec} vcf-consensus {vcf!r} > {output!r}'


    ##################### tabix ####################################################################

    def tabix(self, variants: str | Artifact):
        variants = self._get_artifact(variants)
        self._exec_tabix(variants, output_check=variants.path.with_suffix('.gz.tbi'))

    @simple_exec_and_log
    def _exec_tabix(self, vcfgz: str):
        return f'{self.tools_exec} tabix -fp vcf {vcfgz!r}'
    
    
    ##################### bcftools consensus #######################################################
    
    @wandb_run(job_type=JobType.DATA_GENERATION)
    def bcftools_consensus(self, run: Run | None, variants: str | Artifact, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference, run)
        variants = self._get_artifact(variants, run)
        self.bcftools_index(variants)
        consensus = variants.path.with_suffix('.fa')
        self._exec_bcftools_consesus(reference, self.chroms, variants, consensus, output_check=consensus)
        self._log_artifact(run, consensus)
    
    @simple_exec_and_log
    def _exec_bcftools_consesus(self, fasta: str, regions: list[str], vcf: str, output: str):
        regions = ' '.join([repr(r) for r in regions])
        return f'{self.samtools_exec} faidx {fasta!r} {regions} | {self.tools_exec} bcftools consensus {vcf!r} > {output!r}'
    
    def bcftools_index(self, variants: str | Artifact):
        variants = self._get_artifact(variants)
        self._exec_bcftools_index(variants, output_check=variants.path.with_suffix('.gz.csi'))
        
    @simple_exec_and_log
    def _exec_bcftools_index(self, vcfgz: str):
        return f'{self.tools_exec} bcftools index {vcfgz!r}'
    
    
    ##################### bcftools call ############################################################
    
    @wandb_run(job_type=JobType.DATA_GENERATION)
    def bcftools_call(self, run: Run | None, alignments: str | Artifact, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference, run)
        alignments = self._get_artifact(alignments, run)
        
        self.index_reference(reference)
        alignments_sorted = self.sort_alignments(alignments)
        self.index_alignments(alignments_sorted)
        
        bcf_vcf = alignments.path.with_suffix('.bcf.vcf.gz')
        self._exec_bcftools_call(reference, self.chroms, alignments_sorted, bcf_vcf)
        self._log_artifact(run, bcf_vcf)
        
    @simple_exec_and_log
    def _exec_bcftools_call(self, fasta: str, regions: list[str], bam: str, output: str):
        regions = ','.join(regions)
        args = '-q 0 -Q 0 --max-depth 10000 --min-MQ 0'
        return f'{self.tools_exec} bcftools mpileup {args} -Ou -f {fasta!r} {bam!r} | {self.tools_exec} bcftools call -P 1e-2 -mv -Oz -o {output!r}'
    
    ##################### delly call ###############################################################

    @wandb_run(job_type=JobType.DATA_GENERATION)
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

    @simple_exec_and_log
    def _exec_delly_call(self, fasta: str, bam: str, vcf: str):
        return f'{self.delly_exec} call -g {fasta!r} {bam!r} > {vcf!r}'

    @simple_exec_and_log
    def _exec_bgzip(self, vcf: str, vcfgz: str):
        return f'{self.tools_exec} bgzip -c {vcf!r} > {vcfgz}'


    ##################### samtools faidx ###########################################################

    def index_reference(self, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference)
        self._exec_samtools_faidx(reference, output_check=reference.path.with_suffix('.fa.fai'))

    @simple_exec_and_log
    def _exec_samtools_faidx(self, fasta: str):
        return f'{self.samtools_exec} faidx {fasta!r}'


    ##################### samtools index ###########################################################

    def index_alignments(self, alignments: str | Artifact):
        alignments = self._get_artifact(alignments)
        self._exec_samtools_index(alignments, output_check=alignments.path.with_suffix('.bam.bai'))

    @simple_exec_and_log
    def _exec_samtools_index(self, bam: str):
        return f'{self.samtools_exec} index {bam!r}'


    ##################### samtools sort ############################################################

    def sort_alignments(self, alignments: str | Artifact) -> Path:
        alignments = self._get_artifact(alignments)
        alignments_sorted = alignments.path.with_suffix('.sorted.bam')
        self._exec_samtools_sort(alignments, alignments_sorted, output_check=alignments_sorted)
        return alignments_sorted

    @simple_exec_and_log
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




if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    Fire(DataPipeline)



