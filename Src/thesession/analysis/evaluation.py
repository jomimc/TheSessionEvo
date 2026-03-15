import numpy as np


def calculate_actual_rates(data, dataset):
    """
    Compute the true positive rate and false positive rate from screened alignment results.

    Parameters
    ----------
    data : dict
        Results dictionary containing screened and total positive/negative counts,
        keyed by dataset-prefixed names (e.g. ``"thesession_tunes_screened_positives"``).
    dataset : str
        Dataset name prefix used to look up the correct keys in ``data``.

    Returns
    -------
    tpr : float
        True positive rate: fraction of all positives that MMseqs2 returned
        (i.e. screened positives / total positives).
    fpr : float
        False positive rate: fraction of all negatives that MMseqs2 returned
        (i.e. screened negatives / total negatives).

    Notes
    -----
    "Screened" counts refer to pairs that MMseqs2 actually reported after its
    pre-filter; the denominator is the full set of possible pairs in the dataset.
    These rates therefore reflect MMseqs2's recall/contamination at the retrieval
    stage, before any score threshold is applied.
    """
    # Divide reported pairs by the exhaustive pair counts to get rates
    tpr = data[f"{dataset}_screened_positives"] / data[f"{dataset}_positives"]
    fpr = data[f"{dataset}_screened_negatives"] / data[f"{dataset}_negatives"]
    return tpr, fpr


def get_precision_recall(data, dataset):
    """
    Compute precision and recall curves from ROC data and pair counts.

    Parameters
    ----------
    data : dict
        Results dictionary containing total/positive/negative counts and a
        precomputed ROC curve, keyed by dataset-prefixed names.
    dataset : str
        Dataset name prefix used to look up the correct keys in ``data``.

    Returns
    -------
    precision : np.ndarray
        Precision at each ROC operating point: TP / (TP + FP).
    recall : np.ndarray
        Recall (= TPR) at each ROC operating point: TP / total positives.

    Notes
    -----
    Precision and recall are derived from the ROC curve (fpr, tpr) together
    with the absolute counts of positives (``Nt``) and negatives (``Nf``) among
    MMseqs2-screened pairs (``Mt``, ``Mf``).  This conversion is necessary
    because the ROC curve alone does not encode class prevalence.

    ``N`` (total pairs) is unpacked from the data dict but is not used in the
    calculation; it is retained for potential downstream diagnostics.
    """
    names = ['total', 'positives', 'negatives', 'screened_positives',
             'screened_negatives']
    N, Nt, Nf, Mt, Mf = [data[f"{dataset}_{x}"] for x in names]
    fpr, tpr = data[f"{dataset}_roc"]

    # Convert ROC rates back to absolute TP/FP counts using the screened set
    # sizes, then form the precision and recall at each threshold point
    precision = (tpr * Mt) / (tpr * Mt + fpr * Mf)
    recall = (tpr * Mt) / Nt
    return precision, recall


def calculate_average_precision(data, dataset):
    """
    Compute the area under the precision-recall curve via the step method.

    Parameters
    ----------
    data : dict
        Results dictionary passed through to ``get_precision_recall``.
    dataset : str
        Dataset name prefix passed through to ``get_precision_recall``.

    Returns
    -------
    ap : float
        Average precision: the weighted mean of precisions at each recall
        threshold, approximated as the sum of rectangular steps.

    Notes
    -----
    Uses a right-hand Riemann sum (``precision[1:]`` paired with
    ``recall[1:] - recall[:-1]``), which matches scikit-learn's
    ``average_precision_score`` convention.
    """
    precision, recall = get_precision_recall(data, dataset)
    # Sum rectangular strips under the precision-recall curve (right-hand rule)
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])
