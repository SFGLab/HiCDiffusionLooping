# debug using small bam file
# python hicdifflib/data.py delly_call 4DNFIG85Y8QS:v0
# python hicdifflib/data.py vcf_consensus 4DNFIG85Y8QS.delly.vcf.gz:latest

python hicdifflib/data.py delly_call 4DNFI1GNQM8L:v0
python hicdifflib/data.py vcf_consensus 4DNFI1GNQM8L.delly.vcf.gz:latest
# MA0139.2.meme:v0
python hicdifflib/data.py fimo  # GRCh38-reference-genome
python hicdifflib/data.py fimo --sequence 4DNFI1GNQM8L.delly.vcf.fa:latest
python hicdifflib/data.py pet_pairs \
    --pairs 4DNFI9SL1WSF:v0 \
    --peaks 4DNFIV1N7TLK:v0 \
    --motifs '["fimo_MA0139.2_GRCh38_full_analysis_set_plus_decoy_hla.tsv.gz:latest","fimo_MA0139.2_4DNFI1GNQM8L.delly.vcf.tsv.gz:latest"]'

python hicdifflib/data.py assemble \
    --pet_pairs pet_pairs.csv:latest \
    --sequences '["GRCh38-reference-genome:v0","4DNFI1GNQM8L.delly.vcf.fa:latest"]'
