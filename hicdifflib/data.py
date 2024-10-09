import logging
from io import FileIO
from os import PathLike
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
import subprocess

from fire import Fire
from dataclasses_json import dataclass_json
from wandb.sdk.wandb_run import Run
import wandb


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


@dataclass_json
@dataclass
class DataConfig:
    chroms: list[str] = field(
        default_factory=lambda: [f'chr{i+1}' for i in range(22)] + ['chrX', 'chrY']
        # "$(seq -f 'chr%g' 1 22) chrX chrY"
    )
    data_root: str = './data'
    skip_cache: bool = False
    skip_intermediary: bool = False
    wandb: bool = True
    wandb_project: str = 'HiCDiffusionLooping'
    samtools_exec: str = 'singularity exec samtools.sif samtools'
    delly_exec: str = 'singularity exec delly.sif delly'
    tools_exec: str = 'singularity exec tools.sif'


    def __post_init__(self):
        # self._samtools = SubprocessExecutor(*self.samtools_exec)
        # self._delly = SubprocessExecutor(*self.delly_exec)
        # self._tools = SubprocessExecutor(*self.tools_exec)
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
            run = wandb.init(project=self.wandb_project, name=method.__name__, **wandb_kwargs)
            run.log(self.to_dict())
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


def _log_result(logger, result):
    if result.stdout:
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line:
                logger.info()
    if result.stderr:
        for line in result.stderr.split('\n'):
            line = line.strip()
            if line:
                logger.warning(line.strip())
    if result.returncode:
        logger.error('Exited with %d statuscode', result.returncode)
        raise OSError(result.returncode)
    logger.info('Done')

class DataPipeline(DataConfig):

    @_wandb_run(job_type=JobType.DATA_GENERATION)
    def vcf_consensus(self, run: Run | None, variants: str, reference: str = 'GRCh38-reference-genome:v0'):
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
    def delly_call(self, run: Run | None, alignments: str, reference: str = 'GRCh38-reference-genome:v0'):
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



if __name__ == '__main__':
    Fire(DataPipeline)