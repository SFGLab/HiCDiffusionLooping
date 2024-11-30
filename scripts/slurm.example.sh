#!/bin/bash -l
#SBATCH --job-name "example run"
#SBATCH --account sfglab
#SBATCH --output data/slurm/JOB%j.out
#SBATCH --error data/slurm/JOB%j.err
#SBATCH --time 0-01:00:00
#SBATCH --partition short
#SBATCH --mem 1G
#SBATCH --cpus-per-task 1

source ~/miniforge3/bin/activate hicdiffusion

export NOTIFY_BOT_KEY=""
export WANDB_CACHE_DIR="/mnt/evafs/scratch/shared/imialeshka/.cache/wandb" 
export SINGULARITY_BIND="/mnt/evafs/scratch/shared/imialeshka/hicdata/"

python notify.py "job started"
