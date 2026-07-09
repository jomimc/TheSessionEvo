"""Identifying similar tunes: ROC curves for MMseqs2 tune-family identification (Fig. 1A)."""

import pickle

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from thesession.config import PATH_FIG_DATA
from thesession.io import tune_loader as load_tunes
from thesession.io import savage_loader as savage
from thesession.pipeline.mmseqs import load_mmseqs


###################################################################################################
### IDENTIFYING SIMILAR TUNES (Fig. 1A)

### Runs on: TheSession, Meertens, Savage et al. (English)
### Loads tunes, converts to standard format:
###     "tchroma" is chroma (12-pitch) representation, transposed to C (int 0)
### tchroma is converted to a 12-letter pitch sequence, and saved to fasta
### Runs mmseqs on tune collections (fasta file)
### Loads mmseqs results and calculates roc curves and auc
### Saves data in the format needed for figures
def data_for_identification(redo=False):
    """
    Produce ROC-curve data for Figure 1 (tune-family identification).

    Runs MMseqs2 all-vs-all alignment on three datasets — TheSession,
    Meertens MTC-FS-INST-2.0, and Savage et al. (English) — using the
    12-pitch-class letter encoding.  For each dataset, computes ROC
    curve and AUC treating within-family pairs as positives and
    cross-family pairs as negatives.  Results are pickled to
    ``PATH_FIG_DATA / "fig1_roc_curve_data.pkl"``.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, reprocess all datasets from scratch.
        Default is ``False``.

    Returns
    -------
    None

    Notes
    -----
    The TheSession run uses the full (unfiltered) dataset to maximise
    the number of within-family positives for the ROC curve, then
    applies a post-hoc filter for grace notes, polyphony, and multiple
    voices (but not repeat consistency).
    """
    # Create container for data for figures
    fig_data = {}

    ### TheSession
    print("Running on TheSession data")
    df = load_tunes.load_thesession_tunes(redo=redo)

    # First time this will run and load mmseqs results
    # second time onwards will just load mmseqs results, if 'redo' is not set to True
    dataset = "thesession_tunes"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, redo=redo), dataset)
    fig_data = get_total_positives(df, dataset, 'tune_id', fig_data)
    print(f"  AUC: {fig_data[f'{dataset}_auc']:.3f}")

    ### Meertens
    print("Running on Meertens data")
    df = load_tunes.load_meertens_data(redo=redo)[0]

    dataset = "meertens"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, "ref", redo=redo), dataset, fig_data)
    fig_data = get_total_positives(df, dataset, 'song_id', fig_data)
    print(f"  AUC: {fig_data[f'{dataset}_auc']:.3f}")

    ### Savage et al.
    print("Running on data from Savage et al.")
    df = savage.load_savage_df(full=True, redo=redo)
    df = df.loc[df.Language=='English']

    dataset = "savage_english"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, "ref", redo=redo), dataset, fig_data)
    fig_data = get_total_positives(df, dataset, 'chapter', fig_data)
    print(f"  AUC: {fig_data[f'{dataset}_auc']:.3f}")

    path = PATH_FIG_DATA.joinpath("fig1_roc_curve_data.pkl")
    pickle.dump(fig_data, open(path, 'wb'))
    print(f"Saved figure 1 data to {path}")


### Get overall tpr/fpr, accounting for screening stage
def get_total_positives(df, dataset, x='tune_id', fig_data=None):
    """
    Count the total number of positive and negative pairs in a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with a column ``x`` containing family identifiers.
    dataset : str
        Dataset label used as a key prefix in ``fig_data``.
    x : str, optional
        Column name for family membership.  Default is ``'tune_id'``.
    fig_data : dict or None, optional
        Existing figure-data dict to update.  A new dict is created if
        ``None``.  Default is ``None``.

    Returns
    -------
    fig_data : dict
        Updated dict with keys ``'{dataset}_positives'``,
        ``'{dataset}_negatives'``, and ``'{dataset}_total'``.

    Notes
    -----
    Positives are the number of within-family unordered pairs,
    computed as ``sum(n*(n-1)/2)`` for each family of size ``n``.
    Total is ``len(df)^2`` (all ordered pairs including self-pairs).
    """
    total = len(df)**2
    # Within-family unordered pairs: n*(n-1)/2 per family
    positives = np.sum([n * (n - 1) / 2 for n in df[x].value_counts().values])
    negatives = total - positives
    fig_data[f'{dataset}_positives'] = positives
    fig_data[f'{dataset}_negatives'] = negatives
    fig_data[f'{dataset}_total'] = total
    return fig_data


# Get roc and roc-auc
def get_roc_and_auc(res, dataset, fig_data=None):
    """
    Compute ROC curve and AUC for a set of MMseqs2 alignment hits.

    Parameters
    ----------
    res : pandas.DataFrame
        Annotated hit table with columns ``in_fam`` (bool, True for
        within-family pairs) and ``fident`` (fractional sequence
        identity used as the score).
    dataset : str
        Dataset label used as a key prefix in ``fig_data``.
    fig_data : dict or None, optional
        Existing figure-data dict to update.  A new dict is created if
        ``None``.  Default is ``None``.

    Returns
    -------
    fig_data : dict
        Updated with keys ``'{dataset}_roc'`` (list of [fpr, tpr]),
        ``'{dataset}_auc'``, ``'{dataset}_screened'``,
        ``'{dataset}_screened_positives'``, and
        ``'{dataset}_screened_negatives'``.
    """
    if fig_data is None:
        fig_data = {}
    fpr, tpr, _ = roc_curve(res.in_fam, res.fident)
    auc = roc_auc_score(res.in_fam, res.fident)

    # Save to container
    fig_data[f'{dataset}_roc'] = [fpr, tpr]
    fig_data[f'{dataset}_auc'] = auc
    fig_data[f'{dataset}_screened'] = len(res)
    fig_data[f'{dataset}_screened_positives'] = np.sum(res.in_fam)
    fig_data[f'{dataset}_screened_negatives'] = len(res) - np.sum(res.in_fam)
    return fig_data


### Convert the part ID "{tune_id}_{setting_id}_{part_id}"
### to "{tune_id}_{part_id}" for grouping by same tune/part
def get_uniq(s):
    """
    Extract ``(tune_id, part_index)`` from a part_id string.

    Parameters
    ----------
    s : str
        Part ID string in the format ``"{tune_id}_{setting_id}_{part_no}"``.

    Returns
    -------
    tuple of int
        ``(tune_id, part_no)`` — the first and last underscore-delimited
        fields as integers.

    Notes
    -----
    This is used to group hits by (tune, part) regardless of which
    setting they come from.
    """
    splt = s.split('_')
    return (int(splt[0]), int(splt[-1]))
