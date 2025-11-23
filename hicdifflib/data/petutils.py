import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyranges import PyRanges
from transformers import PreTrainedTokenizer
from Bio.SeqRecord import SeqRecord
from sklearn.neighbors import NearestNeighbors

from hicdifflib.data.base import log_text
from hicdifflib.hicdiffusion import HICDIFFUSION_WINDOW_SIZE

logger = logging.getLogger(__name__)

_to_pyrange_columns = {'chr': 'Chromosome', 'start': 'Start', 'end': 'End', 'strand': 'Strand'}



def check_pair_sequences(
    pet_pairs: pd.DataFrame, 
    chrom_sequences: dict[str, SeqRecord],
    tokenizer: PreTrainedTokenizer,
    max_anchor_tokens: int = 512,
    progress_bar: bool = True,
) -> pd.DataFrame:
    result = []
    for pair_idx in tqdm(pet_pairs.index, disable=not progress_bar):
        row = pet_pairs.loc[pair_idx].copy()
        seq = chrom_sequences.get(row.chr)
        if seq is None:
            continue
        
        # check if any index is higher than a sequence
        if any(x >= len(seq) for x in [row.start_l, row.end_l - 1, row.start_r, row.end_r - 1]):
            continue

        if any(
            len(tokenizer.tokenize(str(seq[anchor]))) > max_anchor_tokens
            for anchor in [slice(row.start_l, row.end_l), slice(row.start_r, row.end_r)]
        ):
            continue

        # bound possible range of window positions which includes both anchors
        min_context_start = max(row.end_r - HICDIFFUSION_WINDOW_SIZE, 0)
        max_context_end = min(
            row.start_l + HICDIFFUSION_WINDOW_SIZE, 
            len(seq), 
            # *([] if self._cooler is None else [self._cooler.chromsizes[row.chr]])
        )
        if (
            min_context_start > row.start_l or
            max_context_end < row.end_r or
            max_context_end - min_context_start < HICDIFFUSION_WINDOW_SIZE
        ):
            continue
        
        row['min_context_start'] = min_context_start
        row['max_context_start'] = max_context_end - HICDIFFUSION_WINDOW_SIZE
        result.append(row.to_frame().T)

    if not result:
        return pd.DataFrame(
            columns=pet_pairs.columns.tolist() + ['min_context_start', 'max_context_start']
        )
    return pd.concat(result, ignore_index=True)


def generate_pet_pairs(
    pairs_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    motifs_df: pd.DataFrame,
    peak_strand_slack: int,
    min_strand_ratio: float,
    anchor_peak_slack: int,
    anchor_peak_min_overlap: int,
    negative_pair_range: int,
    knn_algorithm: str = 'ball_tree', 
    knn_radius: float = 10_000, 
    knn_metric: str = 'euclidean',
    knn_distance_to_drop: float | None = None,
    # knn_distance_to_positive: float | None = None,
):

    anchors_l_df = pairs_df.rename(
        columns={'chr_l': 'chr', 'start_l': 'start', 'end_l': 'end'}
    )
    anchors_l_df['strand'] = '+'
    anchors_r_df = pairs_df.rename(
        columns={'chr_r': 'chr', 'start_r': 'start', 'end_r': 'end'}
    )
    anchors_r_df['strand'] = '-'
    anchors_df = pd.concat([
        anchors_l_df[['pair_index', 'chr', 'start', 'end', 'strand']],
        anchors_r_df[['pair_index', 'chr', 'start', 'end', 'strand']]
    ])
    anchors_df = anchors_df.set_index(['pair_index', 'strand'], drop=False)
    anchors_pr = PyRanges(anchors_df.rename(columns=_to_pyrange_columns))
    logger.info('Pairs anchors:')
    log_text(logger.info, repr(anchors_pr))

    peaks_pr = PyRanges(
        peaks_df.rename(
            columns=dict(index='peak_index', value='peak_value', **_to_pyrange_columns)
        )
    )
    logger.info('Peaks:')
    log_text(logger.info, repr(peaks_pr))

    motifs_pr = PyRanges(motifs_df.rename(columns=dict(score='motif_score', **_to_pyrange_columns)))
    logger.info('Motifs:')
    log_text(logger.info, repr(motifs_pr))

    peaks_stranded_pr = peaks_pr.join(
        other=motifs_pr,
        suffix='_motif',
        report_overlap=True,
        slack=peak_strand_slack,
        apply_strand_suffix=False
    )
    peaks_stranded_pr = peaks_stranded_pr[peaks_stranded_pr.Overlap == peaks_stranded_pr.matched_len-1]
    logger.info('Peaks stranded:')
    log_text(logger.info, repr(peaks_stranded_pr))

    # calculate confidence of peaks' strand
    df = peaks_stranded_pr.as_df()
    df['left_strand'] = (df['Strand'] == '+').astype(int)
    df['right_strand'] = (df['Strand'] == '-').astype(int)
    dfgb = df.groupby('peak_index').agg(
        count=('Strand', 'count'),
        **{
            '+': ('left_strand', 'mean'),
            '-': ('right_strand', 'mean'),
        }
    )
    confident_strands = (
        dfgb
        .reset_index()
        .melt(
            id_vars=['peak_index'],
            value_vars=['+', '-'],
            var_name='Strand',
            value_name='strand_ratio',
        )
    )
    peaks_stranded_pr = PyRanges(
        confident_strands[confident_strands['strand_ratio'] >= min_strand_ratio]
        .merge(
            on=['peak_index', 'Strand'],
            how='left',
            right=(
                peaks_stranded_pr
                .as_df()[['peak_index', 'Chromosome', 'Start', 'End', 'Strand', 'peak_value']]
                .drop_duplicates(['peak_index', 'Strand'])
            )
        )
    )
    logger.info('Confident peaks stranded:')
    log_text(logger.info, repr(peaks_stranded_pr))

    l_pr = PyRanges(
        peaks_stranded_pr
        .as_df()[['Chromosome', 'Start', 'End', 'Strand', 'peak_index', 'peak_value']]
        .query('Strand == "+"')
    )
    r_pr = PyRanges(
        peaks_stranded_pr
        .as_df()[['Chromosome', 'Start', 'End', 'Strand', 'peak_index', 'peak_value']]
        .query('Strand == "-"')
    )
    logger.info('Left and right anchors using stranded peaks')
    log_text(logger.info, repr(l_pr))
    log_text(logger.info, repr(r_pr))

    anchors_filtered_pr = anchors_pr.join(peaks_stranded_pr, suffix='_peak', report_overlap=True, slack=anchor_peak_slack, strandedness='same', how='right')
    anchors_filtered_pr = anchors_filtered_pr[anchors_filtered_pr.Overlap >= anchor_peak_min_overlap]
    logger.info('Filtered anchors:')
    log_text(logger.info, repr(anchors_filtered_pr))

    negatives_df = l_pr.join(r_pr, suffix='_r', slack=negative_pair_range * 2).as_df()
    negatives_df['len_full'] = (negatives_df.End_r - negatives_df.Start)
    negatives_df = negatives_df.query('len_full > 0 & len_full < @negative_pair_range').reset_index(drop=True)
    logger.info('Generated %d negative pairs', len(negatives_df))

    pairs_filtered_df = pairs_df[pairs_df.pair_index.isin(anchors_filtered_pr.pair_index)]
    l_df = (
        anchors_filtered_pr
        .as_df()
        .query('Strand == "+"')[['pair_index', 'peak_index', 'peak_value']]
        .drop_duplicates(['pair_index', 'peak_index'])
    )
    r_df = (
        anchors_filtered_pr
        .as_df()
        .query('Strand == "-"')[['pair_index', 'peak_index', 'peak_value']]
        .drop_duplicates(['pair_index', 'peak_index'])
    )
    positives_df = (
        pairs_filtered_df
        .merge(l_df, on='pair_index')
        .merge(r_df, on='pair_index', suffixes=('_l', '_r'))
    )
    logger.info('Filtered %d positive pairs', len(positives_df))


    if knn_distance_to_drop is None:
        filter = pd.Series(zip(positives_df.peak_index_l, positives_df.peak_index_r))
        df_ids = pd.Series(zip(negatives_df.peak_index, negatives_df.peak_index_r))
        neg_mask = ~df_ids.isin(filter)
    else:
        nbrs = NearestNeighbors(algorithm=knn_algorithm, radius=knn_radius, metric=knn_metric, n_neighbors=10)
        logger.info('Using %s to drop too close negative pairs', repr(nbrs))
        nbrs.fit(negatives_df[['Start', 'End_r']].values)
        distances, indices = nbrs.kneighbors(positives_df[['start_l', 'end_r']].values)
        neg_mask = ~negatives_df.index.isin(np.unique(indices.flatten()[distances.flatten() <= knn_distance_to_drop]))
    
    negatives_df = negatives_df[neg_mask].reset_index(drop=True)
    negatives_df['pet_counts'] = 0
    logger.info('After removal of overlapped generated and positive pairs: negative pairs %d', len(negatives_df))

    dataset_df = pd.concat([
        positives_df[['chr_l', 'start_l', 'end_l', 'start_r', 'end_r', 'pet_counts', 'peak_value_l', 'peak_value_r']]
        .rename(
            columns={'chr_l': 'chr'}
        ),
        negatives_df[['Chromosome', 'Start', 'End', 'Start_r', 'End_r', 'pet_counts', 'peak_value', 'peak_value_r']]
        .rename(
            columns={'Chromosome': 'chr', 'Start': 'start_l', 'End': 'end_l', 'Start_r': 'start_r', 'End_r': 'end_r', 'peak_value': 'peak_value_l'}
        )
    ])
    return dataset_df
