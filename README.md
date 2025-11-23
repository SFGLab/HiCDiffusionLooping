# HiCDiffusionLooping
HiCDiffusion based models for prediction of chromatin loops in human genome

```shell
conda create -n hicdiffusion python=3.11
conda activate hicdiffusion
pip install -r requirements.txt
pip install -e .
```

setup raw data (for W&B only)
```shell
python scripts/log_raw_data.py
python scripts/log_raw_data_v2.py
```

to run main pipeline (result will be saved to ./data)
```shell
export WANDB_CACHE_DIR="./data/wandb"
export SINGULARITY_BIND="./data"
bash scripts/pull_tools.sh # download singularity images with genomics binaries
bash scripts/data_pipeline.sh # preprocess data and generate train and test pairs
python scripts/train_3d_cnn.py encoder_pretrain
python scripts/train_full_model.py full_model # will expect ./data/encoder_pretrain/checkpoint-90000
```

in order to run without W&B:
- `export WANDB_DISABLED=true`
- download manually raw data requried from data/4DN*.tsv
- modify `scripts/data_pipeline.sh` to replace artifact names with full path to files
- modify `scripts/train*.py` to replace `WandbArtifact(...).path` objects to `pathlib.Path('path/to/file')`
