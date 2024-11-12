# export WANDB_CACHE_DIR="/mnt/evafs/scratch/shared/imialeshka/.cache/wandb"
export SINGULARITY_BIND="/mnt/evafs/scratch/shared/imialeshka/hicdata"

function pipeline(){
  python hicdifflib/data.py --data_root $SINGULARITY_BIND $@
}

pipeline delly_call 4DNFI2OEE66L:v0 || exit
pipeline delly_call 4DNFI1GNQM8L:v0 || exit

pipeline vcf_consensus 4DNFI2OEE66L.delly.vcf.gz:latest || exit
pipeline vcf_consensus 4DNFI1GNQM8L.delly.vcf.gz:latest || exit

pipeline fimo --motif MA0139.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA0139.2.meme:v0 --sequence 4DNFI2OEE66L.delly.vcf.fa:latest || exit
pipeline fimo --motif MA0139.2.meme:v0 --sequence 4DNFI1GNQM8L.delly.vcf.fa:latest || exit
