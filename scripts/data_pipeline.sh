# debug using small bam file
# python hicdifflib/data.py delly_call 4DNFIG85Y8QS:v0
# python hicdifflib/data.py vcf_consensus 4DNFIG85Y8QS.delly.vcf.gz:latest

python hicdifflib/data.py delly_call 4DNFI1GNQM8L:v0
python hicdifflib/data.py vcf_consensus 4DNFI1GNQM8L.delly.vcf.gz:latest
# MA0139.2.meme:v0
python hicdifflib/data.py fimo  # GRCh38-reference-genome
python hicdifflib/data.py fimo --sequence 4DNFIG85Y8QS.delly.vcf.fa:v0
