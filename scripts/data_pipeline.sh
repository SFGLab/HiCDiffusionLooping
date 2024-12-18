# export WANDB_CACHE_DIR="/mnt/evafs/scratch/shared/imialeshka/.cache/wandb"
export SINGULARITY_BIND="/mnt/evafs/scratch/shared/imialeshka/hicdata"

function pipeline(){
  python hicdifflib/data.py --data_root $SINGULARITY_BIND $@
}

function to_list(){
    echo "$1" | sed '/^$/d; s/^/"/; s/$/"/' | paste -sd, | sed 's/^/[/; s/$/]/'
}

pipeline delly_call 4DNFI2OEE66L:v0 || exit
pipeline delly_call 4DNFI1GNQM8L:v0 || exit

pipeline vcf_consensus 4DNFI2OEE66L.delly.vcf.gz:latest || exit
pipeline vcf_consensus 4DNFI1GNQM8L.delly.vcf.gz:latest || exit

pipeline fimo --motif MA0139.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA0139.2.meme:v0 --sequence 4DNFI2OEE66L.delly.vcf.fa:latest || exit
pipeline fimo --motif MA0139.2.meme:v0 --sequence 4DNFI1GNQM8L.delly.vcf.fa:latest || exit

pipeline fimo --motif MA1930.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA1930.2.meme:v0 --sequence 4DNFI2OEE66L.delly.vcf.fa:latest || exit
pipeline fimo --motif MA1930.2.meme:v0 --sequence 4DNFI1GNQM8L.delly.vcf.fa:latest || exit

pipeline fimo --motif MA1929.2.meme:v0 --sequence GRCh38-reference-genome:v0 || exit
pipeline fimo --motif MA1929.2.meme:v0 --sequence 4DNFI2OEE66L.delly.vcf.fa:latest || exit
pipeline fimo --motif MA1929.2.meme:v0 --sequence 4DNFI1GNQM8L.delly.vcf.fa:latest || exit


##################### PET PAIRS ################################################
pairs='
4DNFI9SL1WSF:v0
4DNFIA3H6KXY:v0
4DNFIS9CCN6R:v0
4DNFI2BAXOSW:v0
'
peaks='
4DNFIV1N7TLK:v0
4DNFINV9R36U:v0
4DNFIG4HFIT9:v0
4DNFIW1VY2CW:v0
'
motifs='
fimo_MA1929.2_4DNFI1GNQM8L.delly.vcf.tsv.gz:latest
fimo_MA1929.2_4DNFI2OEE66L.delly.vcf.tsv.gz:latest
fimo_MA1929.2_GRCh38_full_analysis_set_plus_decoy_hla.tsv.gz:latest
fimo_MA1930.2_4DNFI1GNQM8L.delly.vcf.tsv.gz:latest
fimo_MA1930.2_4DNFI2OEE66L.delly.vcf.tsv.gz:latest
fimo_MA1930.2_GRCh38_full_analysis_set_plus_decoy_hla.tsv.gz:latest
fimo_MA0139.2_4DNFI2OEE66L.delly.vcf.tsv.gz:latest
fimo_MA0139.2_4DNFI1GNQM8L.delly.vcf.tsv.gz:latest
fimo_MA0139.2_GRCh38_full_analysis_set_plus_decoy_hla.tsv.gz:latest
'
pipeline pet_pairs \
    --pairs $(to_list "$pairs") \
    --peaks $(to_list "$peaks") \
    --motifs $(to_list "$motifs") \
    || exit

# pipeline test_pairs_dataset \
#     --pet_pairs pet_pairs.csv:v1 \
#     --sequences '["GRCh38-reference-genome:v0","4DNFI1GNQM8L.delly.vcf.fa:v0","4DNFI2OEE66L.delly.vcf.fa:v0"]'