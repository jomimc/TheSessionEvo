"""Note substitutions: substitution distance rates and log odds (feeds published Fig. 3)."""

from collections import Counter

import numpy as np
from tqdm import tqdm

from thesession.config import MODES, PATH_FIG_DATA
from thesession.alignment import parts as PA
from thesession.io import savage_loader as savage
from thesession import utils
from thesession.pipeline.mutability import note_prevalence_mutability


###################################################################################################
### Note substitutions

    # Substitution analyses:
    #    Substitution rates + log odds, sub distance (separate by mode, dance, mode and dance, all PID)
def data_for_substitution(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False, mode_alg='exact_pent'):
    """
    Compute and save all data needed for Figure 3 (substitution analysis).

    Runs:

    1. Substitution matrices for all groupings
       (``note_prevalence_mutability``).
    2. Melodic interval distribution (``mint_dist``).
    3. Empirical substitution distance distributions for all pairs and
       per mode (``note_sub_dist``).
    4. Substitution distance for the Savage corpus
       (``note_sub_dist_savage``).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned setting metadata.
    tunes : dict
        Music21-parsed feature dicts.
    df_parts : pandas.DataFrame
        Part-level metadata.
    parts_data : dict
        Part feature dicts.
    res : pandas.DataFrame
        Full annotated hit table.
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.
    mode_alg : str, optional
        Mode-matching algorithm for per-mode breakdowns.
        Default is ``'exact_pent'``.

    Returns
    -------
    None
    """
    print("Computing substitution matrices...")
    _ = note_prevalence_mutability(res0, mismatches, tunes, redo=redo)
    print("Computing melodic interval distribution...")
    _ = mint_dist(tunes, redo=redo)

    path = PATH_FIG_DATA.joinpath(f"sub_dist_all.npy")
    _ = note_sub_dist(res0, mismatches, parts_data, path, alpha=0.5, redo=redo)
    note_sub_dist_savage(redo=redo)

    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    for mode, idx in zip(MODES.keys(), idx_list):
        path = PATH_FIG_DATA.joinpath(f"sub_dist_{mode_alg}_{mode}.npy")
        _ = note_sub_dist(res0.loc[idx], mismatches.loc[idx], parts_data, path, alpha=0.5, redo=redo)



### Get the melodic interval distribution
def mint_dist(tunes, redo=False):
    """
    Compute or load the normalised melodic interval distribution.

    Parameters
    ----------
    tunes : dict
        Mapping from ``setting_id`` to tune feature dict.
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Normalised interval distribution array as returned by
        ``PA.get_mint_dist``.
    """
    path = PATH_FIG_DATA.joinpath(f"mint_dist_tunes.npy")
    if path.exists() and not redo:
        print("  Loading cached melodic interval distribution")
        return np.load(path)
    else:
        print("  Computing melodic interval distribution...")
        mint = PA.get_mint_dist(tunes)
        np.save(path, mint)
        return mint

### Expected substitution distance rate.
### For each tune part, get the expected melodic interval
### distribution one would obtain by shuffling repeatedly.
def get_base_sub_dist_rate(res, mismatches, parts, alpha=0.5):
    """
    Compute the expected substitution distance distribution under a
    null model of random note pairing within each part.

    For each part that appears in the hit table, all pairwise
    absolute pitch differences are tallied (weighted by the number of
    pairs the part is involved in).  This gives the background
    distribution of interval sizes one would expect if substitutions
    were drawn uniformly.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (used for indexing; content not
        used here).
    parts : dict
        Parts feature dicts keyed by ``part_id``.
    alpha : float, optional
        Inverse-frequency weighting exponent (not currently applied;
        included for API consistency).  Default is ``0.5``.

    Returns
    -------
    tot : numpy.ndarray of float, shape (13,)
        Raw count of each interval distance from 1 to 13 semitones
        (index 0 = 1 semitone, index 12 = 13 semitones).
    """
    M = np.arange(1, 14)
    tot = np.zeros(M.size, float)
    parts_dict = Counter(res[['query', 'target']].values.ravel())
    # As far as I can see, the weights do not need to be applied here! ***CHECK
#   weights = utils.inverse_frequency_weights(res, alpha)
    for p, c in parts_dict.items():
        tmidi = parts[p][0][1].astype(int)
        # Get the difference of all notes with all notes, to get
        # all possible melodic intervals
        count = Counter(np.abs(tmidi[:,None] - tmidi[None,:]).ravel())
        # Should this not also be normalized by melody length???
        for k, v in count.items():
            if k in M:
                tot[k-1] += v * c# * w
    return tot


### Empirical substitution distance rate
def note_sub_dist(res, mismatches, parts, path, alpha=0.5, redo=False):
    """
    Compute the empirical substitution distance distribution and its
    log-odds relative to the null model, across PID thresholds.

    For each PID threshold in ``np.arange(0.5, 1, 0.05)``, computes the
    weighted count of observed substitution distances and the log-odds
    ratio against the null (random-pairing) expectation.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (same index as ``res``).
    parts : dict
        Parts feature dicts keyed by ``part_id``.
    path : pathlib.Path
        Cache file path for the result array.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    out : numpy.ndarray, shape (len(pid_list), 2, 13)
        ``out[i, 0]`` is the weighted count of substitution distances at
        PID threshold ``pid_list[i]``; ``out[i, 1]`` is the log-odds
        ratio against the null model.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        pid_list = np.arange(0.5, 1, 0.05)
        X = np.arange(1, 14)
        out = []
        for pid in tqdm(pid_list, desc="  pid thresholds"):
            idx = np.array((res.frac_eq >= pid), bool)

            # Calculate the absolute counts of substitution distances
            # for each tune
            sub_dist = []
            weights = utils.inverse_frequency_weights(res.loc[idx], alpha)
            for sd in mismatches.loc[idx, 'sub_dist']:
                sd_count = Counter(sd)
                sub_dist.append([sd_count.get(i,0) for i in X])

            # Calculate the expected counts of substitution distances
            tot = get_base_sub_dist_rate(res.loc[idx], mismatches.loc[idx], parts, alpha)

            # Sum counts over tunes, with inverse frequency weighting
            Y = np.sum(np.array(sub_dist) * weights[:, None], axis=0)

            # Get the log odds of the ratio of the actual vs expected value
            # log(observed fraction) - log(expected fraction)
            Y2 = np.log(Y / np.sum(Y)) - np.log(tot / tot.sum())

            out.append([Y, Y2])
        out = np.array(out)
        np.save(path, out)
        return out


### Empirical substitution distance rate for the Bronson (British/American)
### and Japanese collections
def note_sub_dist_savage(redo=False):
    """
    Compute or load substitution distance distributions for the Savage
    English and Japanese corpora.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    out : numpy.ndarray, shape (2, 2, 13)
        ``out[0]`` is for English, ``out[1]`` for Japanese.
        Each sub-array has ``[observed_counts, expected_counts]``.
    """
    path = PATH_FIG_DATA.joinpath(f"sub_dist_savage.npy")
    if path.exists() and not redo:
        return np.load(path)
    else:
        df = savage.load_savage_df(full=True, redo=redo)
        languages = ['English', 'Japanese']
        out = []
        for i, l in enumerate(languages):
            idx = df.Language == l
            tot, expected = get_sub_dist_savage(df.loc[idx])
            out.append([tot, expected])
        out = np.array(out)
        np.save(path, out)
        return out


def get_sub_dist_savage(df, redo=False):
    """
    Compute the observed and expected substitution distance distributions
    from manually aligned Savage et al. pairs.

    Each ``PairNo`` in the DataFrame corresponds to a manually aligned
    pair of tune variants.  The function tallies absolute pitch
    differences at substitution positions and at all pairwise note
    positions (for the null expectation).

    Parameters
    ----------
    df : pandas.DataFrame
        Savage subset for one language, with columns ``PairNo``,
        ``tchroma`` (pitch-class array), and ``seq_aligned`` (aligned
        letter string, with ``'-'`` for gaps).
    redo : bool, optional
        Not used; kept for API consistency.

    Returns
    -------
    tot : numpy.ndarray of float, shape (13,)
        Counts of observed substitution distances (1–13 semitones).
    expected : numpy.ndarray of float, shape (13,)
        Counts of all pairwise distances (null model for expected
        distribution).
    """
    M = np.arange(1, 14)
    tot = np.zeros(M.size, float)
    expected = np.zeros(M.size, float)
    pair_list = np.array(sorted(df['PairNo'].unique()))
    for pair in pair_list:
        idx = df.loc[df['PairNo'] == pair].index
        if len(idx) != 2:
            continue
        tc1, tc2 = df.loc[idx, 'tchroma']
        al1, al2 = [np.array(list(x)) for x in df.loc[idx, 'seq_aligned']]

        if len(al1) != len(al2):
            print(f"Error in manual alignment for {pair}")
            continue

        # Remove indels
        # Keep only positions where the corresponding alignment column is not a gap
        idx1 = np.where(al1 != '-')[0]
        idx2 = np.where(al2 != '-')[0]
        tc1 = tc1[al2[idx1] != '-']
        tc2 = tc2[al1[idx2] != '-']

        # Get substitutions
        sub_idx = np.where(tc1 != tc2)[0]
        sub_dist = np.abs(tc1[sub_idx] - tc2[sub_idx])
        for d, c in Counter(sub_dist).items():
            if d in M:
                tot[d-1] += c

        # Get aligned notes
        # Compute all pairwise absolute distances within each sequence for the null
        for tc in [tc1, tc2]:
            count = Counter(np.abs(tc[:,None] - tc[None,:]).ravel())
            for d, c in count.items():
                if d in M:
                    expected[d-1] += c

    return tot, expected
