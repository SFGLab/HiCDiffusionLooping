echo ${SINGULARITY_BIND?}
echo ${TRIOS_ROOT?}

function pipeline(){
  python hicdifflib/data/pipeline.py --data_root $SINGULARITY_BIND $@
}

function to_list(){
    echo "$1" | sed '/^$/d; s/^/"/; s/$/"/' | paste -sd, | sed 's/^/[/; s/$/]/'
}

for s in GM19238 GM19239 GM19240 HG00512 HG00513 HG00514 HG00731 HG00732 HG00733; 
do 
    echo "*******************" $s "****************";
    for m in MA0139.2.meme:v0 MA1930.2.meme:v0 MA1929.2.meme:v0;
    do
        echo "*******************" $m "****************";
        python notify.py "$s $m"
        pipeline fimo --motif ${m} --sequence $TRIOS_ROOT/diploid/${s}_maternal/${s}_mat.fa || exit;
        pipeline fimo --motif ${m} --sequence $TRIOS_ROOT/diploid/${s}_paternal/${s}_pat.fa || exit;
        pipeline fimo --motif ${m} --sequence $TRIOS_ROOT/haploid/${s}_haploid_chrbased.fa || exit;
    done;
done;


for s in GM19238 GM19239 GM19240 HG00512 HG00513 HG00514 HG00731 HG00732 HG00733; 
do 
    echo "*******************" $s "****************";
    for ht in pat mat;
    do
        echo "*******************" $s $ht "****************";
        HT=$(echo $ht | tr a-z A-Z)
        pairs="
$TRIOS_ROOT/hichip/${s}_${HT}_SG/final_output/${s}_${HT}.bedpe"
        peaks="
$TRIOS_ROOT/hichip/${s}_${HT}_SG/final_output/${s}_${HT}_peaks.narrowPeak"
        motifs="
fimo_MA1929.2_${s}_${ht}.tsv.gz:latest
fimo_MA1930.2_${s}_${ht}.tsv.gz:latest
fimo_MA0139.2_${s}_${ht}.tsv.gz:latest"
        pipeline pet_pairs \
            --pairs $(to_list "$pairs") \
            --peaks $(to_list "$peaks") \
            --motifs $(to_list "$motifs") \
            --name ${s}_${ht} \
            --strict_anchors False \
            --knn_distance_to_drop 10000 \
            --peak_score_idx 4 \
            --pairs_header True \
            --pairs_extra_columns "['count','expected','fdr','ClusterLabel','ClusterSize','ClusterType','ClusterNegLog10P','ClusterSummit']" \
            --pairs_counts_column count \
            || echo "*******************" FAILED $s $ht "****************";
    done;
    echo "*******************" $s hap "****************";
    pairs="
$TRIOS_ROOT/hichip/${s}_SG/final_output/${s}.bedpe"
    peaks="
$TRIOS_ROOT/hichip/${s}_SG/final_output/${s}_peaks.narrowPeak"
    motifs="
fimo_MA1929.2_${s}_haploid_chrbased.tsv.gz:latest
fimo_MA1930.2_${s}_haploid_chrbased.tsv.gz:latest
fimo_MA0139.2_${s}_haploid_chrbased.tsv.gz:latest"
    pipeline pet_pairs \
        --pairs $(to_list "$pairs") \
        --peaks $(to_list "$peaks") \
        --motifs $(to_list "$motifs") \
        --name ${s}_hap \
        --strict_anchors False \
        --knn_distance_to_drop 10000 \
        --peak_score_idx 4 \
        --pairs_header True \
        --pairs_extra_columns "['count','expected','fdr','ClusterLabel','ClusterSize','ClusterType','ClusterNegLog10P','ClusterSummit']" \
        --pairs_counts_column count \
        || echo "*******************" FAILED $s hap "****************";
done;



for s in GM19238 GM19239 GM19240 HG00512 HG00513 HG00514 HG00731 HG00732 HG00733; 
do 
    for subset in train eval test;
    do
        echo "*******************" $s pat $subset "****************";
        pipeline --chroms_set $subset filter_pairs \
            --pet_pairs ${s}_pat_pairs.csv:latest \
            --sequence $TRIOS_ROOT/diploid/${s}_paternal/${s}_pat.fa \
            --reference $TRIOS_ROOT/diploid/${s}_paternal/${s}_pat.fa \
            --tokenizer_name None \
            --name ${subset}_${s}_pat  \
            || echo fail
        echo "*******************" $s mat $subset "****************";
        pipeline --chroms_set $subset filter_pairs \
            --pet_pairs ${s}_mat_pairs.csv:latest \
            --sequence $TRIOS_ROOT/diploid/${s}_maternal/${s}_mat.fa \
            --reference $TRIOS_ROOT/diploid/${s}_maternal/${s}_mat.fa \
            --tokenizer_name None \
            --name ${subset}_${s}_mat  \
            || echo fail
        echo "*******************" $s hap $subset "****************";
        pipeline --chroms_set $subset filter_pairs \
            --pet_pairs ${s}_hap_pairs.csv:latest \
            --sequence $TRIOS_ROOT/haploid/${s}_haploid_chrbased.fa \
            --reference $TRIOS_ROOT/haploid/${s}_haploid_chrbased.fa \
            --tokenizer_name None \
            --name ${subset}_${s}_hap  \
            || echo fail
    done;
done;

