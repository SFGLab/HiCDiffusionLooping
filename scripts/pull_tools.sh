[ ! -f tools.sif ] && \
    singularity pull tools.sif library://m10an/genomics/tools || \
    echo '***skipping tools download***';

[ ! -f samtools.sif ] && \
    singularity pull samtools.sif library://millironx/default/samtools || \
    echo '***skipping samtools download***';

[ ! -f delly.sif ] && \
    wget -O delly.sif https://github.com/dellytools/delly/releases/download/v1.2.6/delly_v1.2.6.sif || \
    echo '***skipping delly download***';
