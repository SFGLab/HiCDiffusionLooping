import logging

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred, thr = None, thr_search_step=0.01):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits[:, 0]))

    thr_best = 0
    f1_best = 0
    if thr is None:
        try:
            for thr_test in np.arange(thr_search_step, 1, thr_search_step):
                f1_test = f1_score(labels, (probabilities >= thr_test).astype(int))
                if f1_test > f1_best:
                    f1_best = f1_test
                    thr_best = thr_test
            thr = thr_best
        except Exception as e:
            thr = 0.5
            logger.warning(str(e))
        
    predictions = (probabilities >= thr).astype(int)

    try:
        f1 = f1_score(labels, predictions)
    except Exception as e:
        f1 = np.nan
        logger.warning(str(e))

    try:
        precision = precision_score(labels, predictions)
    except Exception as e:
        precision = np.nan
        logger.warning(str(e))

    try:
        recall = recall_score(labels, predictions)
    except Exception as e:
        recall = np.nan
        logger.warning(str(e))

    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except Exception as e:
        roc_auc = np.nan
        logger.warning(str(e))

    try:
        pr_auc = average_precision_score(labels, probabilities)
    except Exception as e:
        pr_auc = np.nan
        logger.warning(str(e))

    return {
        "threshold": thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": pr_auc
    }
