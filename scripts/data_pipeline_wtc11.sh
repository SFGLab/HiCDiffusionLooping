# export WANDB_CACHE_DIR="/mnt/evafs/scratch/shared/imialeshka/.cache/wandb"
export SINGULARITY_BIND="/mnt/evafs/scratch/shared/imialeshka/hicdata"

function pipeline(){
  python hicdifflib/data.py --data_root $SINGULARITY_BIND $@
}

function to_list(){
    echo "$1" | sed '/^$/d; s/^/"/; s/$/"/' | paste -sd, | sed 's/^/[/; s/$/]/'
}

pipeline delly_call wtc11_rep1_4DNFIUE5AYN5:v0 || exit
pipeline delly_call wtc11_rep2_4DNFILITYTOS:v0 || exit

pipeline vcf_consensus wtc11_rep1_4DNFIUE5AYN5.delly.vcf.gz:latest || exit
pipeline vcf_consensus wtc11_rep2_4DNFILITYTOS.delly.vcf.gz:latest || exit

pipeline fimo --motif MA0139.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA0139.2.meme:v0 --sequence wtc11_rep1_4DNFIUE5AYN5.delly.vcf.fa:latest || exit
pipeline fimo --motif MA0139.2.meme:v0 --sequence wtc11_rep2_4DNFILITYTOS.delly.vcf.fa:latest || exit

pipeline fimo --motif MA1930.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA1930.2.meme:v0 --sequence wtc11_rep1_4DNFIUE5AYN5.delly.vcf.fa:latest || exit
pipeline fimo --motif MA1930.2.meme:v0 --sequence wtc11_rep2_4DNFILITYTOS.delly.vcf.fa:latest || exit

pipeline fimo --motif MA1929.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA1929.2.meme:v0 --sequence wtc11_rep1_4DNFIUE5AYN5.delly.vcf.fa:latest || exit
pipeline fimo --motif MA1929.2.meme:v0 --sequence wtc11_rep2_4DNFILITYTOS.delly.vcf.fa:latest || exit
