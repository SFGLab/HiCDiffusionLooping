function pipeline(){
  python hicdifflib/data/pipeline.py --data_root $SINGULARITY_BIND $@
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
    --name gm12878 \
    || exit


##################### FILTER PAIRS #############################################

pipeline --chroms_set "train" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence GRCh38-reference-genome:v0 \
    --max_anchor_tokens 510 \
    --name train_hg38_gm12878 \
    || exit

pipeline --chroms_set "train" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence 4DNFI1GNQM8L.delly.vcf.fa:v0 \
    --min_chrom_length_match 0.2 \
    --max_anchor_tokens 510 \
    --name train_4DNFI1GNQM8L_gm12878 \
    || exit

pipeline --chroms_set "train" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence 4DNFI2OEE66L.delly.vcf.fa:v0 \
    --min_chrom_length_match 0.2 \
    --max_anchor_tokens 510 \
    --name train_4DNFI2OEE66L_gm12878 \
    || exit

pipeline --chroms_set "eval" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence GRCh38-reference-genome:v0 \
    --max_anchor_tokens 510 \
    --name eval_hg38_gm12878 \
    || exit

pipeline --chroms_set "eval" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence 4DNFI1GNQM8L.delly.vcf.fa:v0 \
    --min_chrom_length_match 0.95 \
    --max_anchor_tokens 510 \
    --name eval_4DNFI1GNQM8L_gm12878 \
    || exit

pipeline --chroms_set "eval" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence 4DNFI2OEE66L.delly.vcf.fa:v0 \
    --min_chrom_length_match 0.95 \
    --max_anchor_tokens 510 \
    --name eval_4DNFI2OEE66L_gm12878  \
    || exit

pipeline --chroms_set "test" filter_pairs \
    --pet_pairs gm12878_pairs.csv:latest \
    --sequence GRCh38-reference-genome:v0 \
    --max_anchor_tokens 510 \
    --name test_hg38_gm12878  \
    || exit
