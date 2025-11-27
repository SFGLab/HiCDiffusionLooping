import os
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
import subprocess

import wandb
from wandb.sdk.wandb_run import Run
from dataclasses_json import dataclass_json


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
    
    def __eq__(self, value):
        return str(self) == str(value)

    @property
    def path(self) -> Path:
        return Path(str(self))
    
    @property
    def path_no_suffix(self) -> Path:
        return self.path.with_suffix('')


class LocalArtifact(Artifact):
    def __init__(self, path: str) -> None:
        self._path = str(path)
    
    def __str__(self) -> str:
        return self._path


class WandbArtifact(Artifact):
    def __init__(self, name: str, config: DataConfig, run: Run | None = None) -> None:
        self._artifact: wandb.Artifact = (
            run.use_artifact(name) if run else config._api.artifact(name)
        )
        self._root_dir = self._artifact.download(
            root=config.data_root,
            skip_cache=config.skip_cache,
        )
        self._entry_path = list(self._artifact.manifest.entries.values()).pop().path
    
    def __str__(self) -> str:
        return os.path.join(self._root_dir, self._entry_path)


def simple_exec_and_log(cmd_builder: callable) -> callable:
    """Wraps BasePipeline method which builds cmd string
    """

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
        log_result(logger, result)
    
    return decorated_method


def wandb_run(**wandb_kwargs):
    """Wraps BasePipeline method with wandb loggin and creds
    """

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


def log_text(logg_fn, text):
    for line in text.split('\n'):
        line = line.strip()
        if line:
            logg_fn(line)


def log_result(logger, result):
    if result.stdout:
        log_text(logger.info, result.stdout)
    if result.stderr:
        log_text(logger.warning, result.stderr)
    if result.returncode:
        logger.error('Exited with %d statuscode', result.returncode)
        raise OSError(result.returncode)
    logger.info('Done')
