import numpy as np


### Compute actual (post-screening) TPR and FPR from fig1 data dict
def calculate_actual_rates(data, dataset):
    tpr = data[f"{dataset}_screened_positives"] / data[f"{dataset}_positives"]
    fpr = data[f"{dataset}_screened_negatives"] / data[f"{dataset}_negatives"]
    return tpr, fpr


def get_precision_recall(data, dataset):
    names = ['total', 'positives', 'negatives', 'screened_positives',
             'screened_negatives']
    N, Nt, Nf, Mt, Mf = [data[f"{dataset}_{x}"] for x in names]
    fpr, tpr = data[f"{dataset}_roc"]
    precision = (tpr * Mt) / (tpr * Mt + fpr * Mf)
    recall = (tpr * Mt) / Nt
    return precision, recall


def calculate_average_precision(data, dataset):
    precision, recall = get_precision_recall(data, dataset)
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])
