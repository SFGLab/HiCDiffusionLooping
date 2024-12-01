from hicdifflib.nn import logger


import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits[:, 0]))
    predictions = (probabilities >= 0.5).astype(int)

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
        f1 = f1_score(labels, predictions)
    except Exception as e:
        f1 = np.nan
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": pr_auc
    }
