import logging
from io import FileIO
from os import PathLike
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from subprocess import Popen, PIPE

from fire import Fire
from dataclasses_json import dataclass_json
from wandb.sdk.wandb_run import Run
import wandb


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.StreamHandler(),
    # ]
)

logger = logging.getLogger(__name__)


class WandbLoggingHandler(logging.Handler):
    def __init__(self, run: Run, level=0) -> None:
        super().__init__(level)
        self._run = run

    def emit(self, record):
        message = self.format(record)
        # self._run.log({'log_message': message})



@dataclass_json
@dataclass
class DataConfig:
    data_root: str = './data'
    skip_cache: bool = False
    skip_intermediary: bool = False
    wandb: bool = True
    wandb_project: str = 'HiCDiffusionLooping'
    samtools_exec: list[str] = field(
        default_factory=lambda: ['singularity', 'exec', 'samtools.sif', 'samtools']
    )
    delly_exec: list[str] = field(
        default_factory=lambda: ['singularity', 'exec', 'delly.sif', 'delly']
    )
    tools_exec: list[str] = field(
        default_factory=lambda: ['singularity', 'exec', 'tools.sif']
    )


    def __post_init__(self):
        self._samtools = SubprocessExecutor(*self.samtools_exec)
        self._delly = SubprocessExecutor(*self.delly_exec)
        self._tools = SubprocessExecutor(*self.tools_exec)
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


class SubprocessExecutor:
    def __init__(self, *cmd) -> None:
        self._cmd = cmd

    @staticmethod
    def _decode(s):
        return s.decode(errors='ignore') if isinstance(s, bytes) else s

    def run(self, *cmd, output: FileIO = PIPE, log_output: bool = False) -> str | None:
        cmd_ = [str(x) for x in self._cmd + cmd]
        logger.info(' '.join(cmd_) + (f' -> ({output.name!r},{output.mode!r})' if output is not PIPE else ''))

        process = Popen(cmd_, stdout=output, stderr=PIPE, text=True, bufsize=1)
        with process.stderr:
            for line in iter(process.stderr.readline, ''):
                logger.info(line.strip())
            
        out = None
        if output is PIPE:
            with process.stdout:
                out = process.stdout.read()
        process.wait()
        if log_output:
            logger.info(out)
        if output is PIPE:
            return out


def _wandb_run(**wandb_kwargs):

    def decorate_run(method: callable) -> callable:

        def decorated_method(self: DataConfig, *args, **kwargs) -> Any:
            if not self.wandb:
                return method(self, None, *args, **kwargs)
            run = wandb.init(project=self.wandb_project, name=method.__name__, **wandb_kwargs)
            # handler = WandbLoggingHandler(run)
            # logger.addHandler(handler)
            run.log(self.to_dict())
            result = method(self, run, *args, **kwargs)
            # logger.removeHandler(handler)
            run.finish()
            return result
        
        return decorated_method
        
    return decorate_run


class DataPipeline(DataConfig):

    def test_output(self):
        self._samtools.run('--help', log_output=True)
        with open('data/out.txt', 'w') as f:
            self._samtools.run('--help', output=f)

        

    @_wandb_run(job_type=JobType.DATA_GENERATION)
    def delly_call(self, run: Run | None, alignments: str, reference: str = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference, run)
        alignments: Artifact = self._get_artifact(alignments, run)

        self.index_reference(reference)
        alignments_sorted = self.sort_alignments(alignments)
        self.index_alignments(alignments_sorted)

        delly_vcf = alignments.path.with_suffix('.delly.vcf')
        if not delly_vcf.exists() or self.skip_intermediary:
            with open(delly_vcf, 'w') as f:
                self._delly.run('call', '-g', reference, alignments_sorted, output=f)

        delly_vcfgz = delly_vcf.with_suffix('.vcf.gz')
        if not delly_vcfgz.exists() or self.skip_intermediary:
            with open(delly_vcfgz, 'w') as f:
                self._tools.run('bgzip', '-c', delly_vcf, output=f)
        
        if self.wandb:
            result = wandb.Artifact(
                name=delly_vcfgz.name,
                type="dataset",
            )
            result.add_file(str(delly_vcfgz))
            run.log_artifact(result)
        
        # self._tools.run('tabix', '-p', 'vcf', delly_vcfgz, log_output=True)



    def index_reference(self, reference: str | Artifact = 'GRCh38-reference-genome:v0'):
        reference = self._get_artifact(reference)
        if not reference.path.with_suffix('.fa.fai').exists() or self.skip_intermediary:
            self._samtools.run('faidx', reference, log_output=True)

    
    def index_alignments(self, alignments: str | Artifact):
        alignments = self._get_artifact(alignments)
        if not alignments.path.with_suffix('.bam.bai').exists() or self.skip_intermediary:
            self._samtools.run('index', alignments, log_output=True)


    def sort_alignments(self, alignments: str | Artifact) -> Path:
        alignments = self._get_artifact(alignments)
        alignments_sorted = alignments.path.with_suffix('.sorted.bam')
        if not alignments_sorted.exists() or self.skip_intermediary:
            self._samtools.run('sort', '-o', alignments_sorted, alignments)
        return alignments_sorted


    def _get_artifact(self, artifact: str | Artifact, run: Run | None = None) -> Artifact:
        if isinstance(artifact, Artifact):
            return artifact
        if isinstance(artifact, Path) or ':' not in str(artifact):
            return LocalArtifact(artifact)
        return WandbArtifact(artifact, self, run)
        


if __name__ == '__main__':
    Fire(DataPipeline)