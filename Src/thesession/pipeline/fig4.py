"""Sequence position (Fig. 4): within-bar and across-bar substitution rates."""

from collections import Counter
import pickle

import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from tqdm import tqdm

from thesession.config import DANCE_LIST, METER_LIST, MODES, PATH_FIG_DATA, SUBDIV_DANCE, SUBDIV_METER
from thesession import utils


###################################################################################################
### Sequence position (Fig. 4)

    # Run analyses for Fig 4:
    #    Within-measure / across-measure rates, hierarchy and prevalence, (separate by mode, dance, mode and dance, all PID)
    #    covariance and repetition (separate by mode, dance, mode and dance, all and most common 100 tunes)
def data_for_fig4(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False):
    """
    Compute and save all data needed for Figure 4 (positional analysis).

    Runs:

    1. Bar-level substitution rates by meter, dance, and mode
       (``bar_rate``).
    2. Within-bar positional substitution rates by meter and dance
       (``bar_pos_rate``).
    3. Onset histograms by meter (``onset_histograms``).
    4. Correlations between positional substitution rate and metrical
       hierarchy / onset stability (``bar_pos_rate_corr``).

    Parameters
    ----------
    df : pandas.DataFrame
        Setting metadata.
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

    Returns
    -------
    None
    """
    print("Computing bar substitution rates...")
    _ = bar_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=redo)
    print("Computing bar position substitution rates...")
    _ = bar_pos_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=redo)
    print("Computing onset histograms...")
    _ = onset_histograms(res0, parts_data, redo=redo)
    print("Computing bar position rate correlations...")
    _ = bar_pos_rate_corr(redo=redo)


### Calculate the average substitution rate in a measure, as a function
### of the position of the measure in the part.
### For many different groups of tune parts.
def bar_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False):
    """
    Compute per-bar substitution rates for all standard groupings.

    Restricts to the four main dance types (reel, jig, polka, hornpipe)
    which are known to have the 8-bar part structure, then computes
    bar-level substitution rates for:

    * All pairs combined.
    * Each of the three most common meters (4/4, 6/8, 2/4).
    * Each dance type.
    * Each mode (under ``mode_alg``).

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    mode_alg : str, optional
        Mode-compatibility algorithm.  Default is ``'exact_pent'``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    rate_dict : dict
        Keys are grouping labels (``'all'``, meter strings, dance names,
        mode names); values are arrays of shape
        ``(len(pid_list), 4, nbars)`` — mean, std, and 95% CI bounds.
    """
    pid_list = np.arange(0.5, 1, 0.05)
    dance_list = ['reel', 'jig', 'polka', 'hornpipe']
    rate_dict = {}

    # Only look at dances that are known to have the 8-bar structure
    idx = np.array(res0.target_dance.isin(dance_list) & res0.query_dance.isin(dance_list), bool)
    res0 = res0.loc[idx]
    mismatches = mismatches.loc[idx]

    # bar rate: all
    path = PATH_FIG_DATA.joinpath("bar_rate-all.npy")
    rate_dict['all'] = get_bar_rate(res0, mismatches, pid_list, path, alpha=alpha, redo=redo)
    print(f"{len(res0)} pairs used for All")

    # bar rate: meter
    # Only run on 4/4, 2/4 and 6/8
    for meter in METER_LIST[:3]:
        idx = np.array((res0.target_meter==meter)&(res0.query_meter==meter), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{meter.replace('/', '_')}.npy")
        rate_dict[meter] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {meter}")

    # bar rate: dance
    for dance in dance_list:
        idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{dance}.npy")
        rate_dict[dance] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {dance}")

    # bar rate: mode
    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    for mode, idx in zip(MODES.keys(), idx_list):
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{mode}.npy")
        rate_dict[mode] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {mode}")

    return rate_dict


### Calculate the average substitution rate in a measure, as a function
### of the position of the measure in the part.
### Do this for different PID thresholds.
def get_bar_rate(res, mismatches, pid_list, path, alpha=0.5, redo=False):
    """
    Compute mean bar substitution rate with bootstrapped confidence
    intervals across PID thresholds.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table subset.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (same index as ``res``).
    pid_list : array-like of float
        PID thresholds to evaluate.
    path : pathlib.Path
        Cache file path.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    rate_stats : numpy.ndarray, shape (len(pid_list), 4, nbars)
        For each PID threshold: ``[mean, std, 2.5th_percentile,
        97.5th_percentile]`` of the per-bar substitution rate across
        bootstrap resamples.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        ci = [0.025, 0.975]
        rate_stats = []
        for pid in tqdm(pid_list, desc="  pid thresholds"):
            idx = np.array(res.fident > pid, bool)
            weights = utils.inverse_frequency_weights(res.loc[idx], alpha)
            sub_rate_all = get_bar_subrate(mismatches.loc[idx], max_bar=8)
            Y, Ysample = get_bar_subrate_stats(sub_rate_all, weights)
            Ys = np.std(Ysample, axis=0)
            # Save the mean, standard deviation, and the 95% CI
            rate_stats.append([Y, Ys] + list(np.quantile(Ysample, ci, axis=0)))
        rate_stats = np.array(rate_stats)
        np.save(path, rate_stats)
        return rate_stats


### Get bar substitution rate
def get_bar_subrate(mismatches, max_bar=8):
    """
    Compute the per-bar substitution rate for each pair in a mismatch
    table.

    Rates are expressed in bar-fraction units (substitutions per total
    bar duration) rather than raw counts, so that pairs with finer grid
    resolutions are not artificially inflated.

    Parameters
    ----------
    mismatches : pandas.DataFrame
        Sub-table of per-pair alignment statistics with columns
        ``'sub_bar'`` and ``'grid_per_bar'``.
    max_bar : int, optional
        Number of bars to include (bars beyond this index are ignored).
        Default is ``8``.

    Returns
    -------
    sub_rate_all : numpy.ndarray, shape (n_pairs, max_bar)
        Per-bar substitution rate for each pair.
    """
    sub_rate_all = []
    # Calculate substitution rate per bar for each tune
    for sub_bar, grid_per_bar in zip(*mismatches[['sub_bar', 'grid_per_bar']].values.T):
        sub_rate = np.zeros(max_bar)
        for b in sub_bar[sub_bar < max_bar]:
            # The number of substitutions depends on how coarsely the bar
            # has been discretized.
            # Thus, we measure rates in terms of bar fraction,
            # i.e. a rate of 0.5 means notes equal to have the total bar duration are substituted
            sub_rate[b] += 1 / grid_per_bar
        sub_rate_all.append(sub_rate)
    return np.array(sub_rate_all)


### Calculate the substitution rate as a function of the position within a measure.
### For many different groups of tune parts.
def bar_pos_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False):
    """
    Compute within-bar positional substitution rates for all meter and
    dance groupings.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    mode_alg : str, optional
        Mode-compatibility algorithm.  Default is ``'exact_pent'``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    rate_dict : dict
        Keys are meter or dance label strings; values are arrays of
        shape ``(len(pid_list), 4, subdivision)`` returned by
        ``get_bar_pos_rate``.
    """
    pid_list = np.arange(0.5, 1, 0.05)
    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    rate_dict = {}

    # bar pos rate: meter
    for meter in METER_LIST:
        sd = SUBDIV_METER[meter]
        idx = np.array((res0.target_meter==meter)&(res0.query_meter==meter), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{meter.replace('/', '_')}.npy")
        rate_dict[meter] = get_bar_pos_rate(res0.loc[idx], mismatches.loc[idx], sd, pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {meter}")

    # bar pos rate: dance
    for dance in DANCE_LIST:
        sd = SUBDIV_DANCE[dance]
        idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{dance}.npy")
        rate_dict[dance] = get_bar_pos_rate(res0.loc[idx], mismatches.loc[idx], sd, pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {dance}")

    return rate_dict


### Calculate the substitution rate as a function of the position within a measure.
### For different PID thresholds.
def get_bar_pos_rate(res, mismatches, subdivision, pid_list, path, alpha=0.5, redo=False):
    """
    Compute within-bar positional substitution rates with bootstrapped
    confidence intervals across PID thresholds.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table subset.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    subdivision : int
        Number of positions per bar (from ``SUBDIV_METER`` or
        ``SUBDIV_DANCE``).
    pid_list : array-like of float
        PID thresholds to evaluate.
    path : pathlib.Path
        Cache file path.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    numpy.ndarray, shape (len(pid_list), 4, subdivision)
        For each PID threshold: ``[mean, std, 2.5th_percentile,
        97.5th_percentile]`` of the positional substitution rate.

    Notes
    -----
    ``sub_pos`` values are in units of bar fraction [0, 1).  They are
    converted to bar-subdivision units by multiplying by
    ``subdivision`` before binning.  Rounding handles floating-point
    precision errors.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        ci = [0.025, 0.975]
        rate_stats = []
        for pid in tqdm(pid_list, desc="  pid thresholds"):
            idx = np.array(res.fident > pid, bool)
            weights = utils.inverse_frequency_weights(res.loc[idx], alpha)
            # Calculate substitution rate per bar for each tune
            sub_rate_all = []
            for sub_pos, nbar in zip(mismatches.loc[idx, 'sub_pos'], res.loc[idx, 'nbars']):
                # sub_pos is given in units of fraction of total bar duration
                # This converts it to units of (usually eighth notes, but can be finer grained)
                # Rounding takes care of floating point errors (e.g. 1.99999)
                sub_pos_count = Counter(np.round(sub_pos * subdivision, 1))

                # Only include integers, and divide by the number of bars
                # to get rate in units of substitutions per bar
                sub_rate = np.array([sub_pos_count[i] / nbar for i in range(subdivision)])
                sub_rate_all.append(sub_rate)

            Y, Ysample = get_bar_subrate_stats(np.array(sub_rate_all), weights)
            Ys = np.std(Ysample, axis=0)
            # Save the mean, standard deviation, and the 95% CI
            rate_stats.append([Y, Ys] + list(np.quantile(Ysample, ci, axis=0)))
        np.save(path, rate_stats)
        return np.array(rate_stats)


### Bootstrap substitution rates to get confidence intervals
def get_bar_subrate_stats(sub_rate_all, weights, nrep=1000):
    """
    Compute the weighted mean and bootstrapped distribution of
    per-bar substitution rates.

    Parameters
    ----------
    sub_rate_all : numpy.ndarray, shape (n_pairs, nbars)
        Per-pair per-bar substitution rates.
    weights : numpy.ndarray of float, shape (n_pairs,)
        Inverse-frequency weights for each pair.
    nrep : int, optional
        Number of bootstrap replicates.  Default is ``1000``.

    Returns
    -------
    Y : numpy.ndarray of float, shape (nbars,)
        Weighted mean substitution rate per bar.
    Ysample : numpy.ndarray of float, shape (nrep, nbars)
        Bootstrap distribution of the weighted mean.

    Notes
    -----
    Bootstrap resamples are drawn with replacement at the pair level.
    The weights of the resampled pairs are renormalised within each
    replicate so that the weighted mean is unbiased.
    """
    Ysample = []

    # Calculate the weighted mean
    w = weights / weights.sum()
    Y = np.sum(sub_rate_all * w[:,None], axis=0)

    # Calculate errors from bootstrapping
    idx = np.arange(w.size)
    for _ in range(nrep):
        sample = np.random.choice(idx, size=w.size, replace=True)
        Ysample.append(np.sum(sub_rate_all[sample] * w[sample,None] / np.sum(w[sample]), axis=0))
    Ysample = np.array(Ysample)
    return Y, Ysample


### Get the probability of an onset occurring at a point within a measure
def onset_histograms(res0, parts_data, redo=False):
    """
    Compute onset-position histograms (probability of a note onset at
    each within-bar position) for each meter.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table (used to group by meter).
    parts_data : dict
        Parts feature dicts keyed by ``part_id``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    hist_stats : dict
        Keys are meter strings; values are lists
        ``[mean, std, lower_CI, upper_CI]`` of shape
        ``(subdivision,)`` each.
    """
    path = PATH_FIG_DATA.joinpath("onset_histograms.pkl")
    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        ci = [0.025, 0.975]
        hist_stats = {}
        for meter in METER_LIST:
            idx = (res0.target_meter==meter)&(res0.query_meter==meter)
            hist = get_onset_histograms(res0.loc[idx], parts_data, SUBDIV_METER[meter])
            # Normalise each row to a probability distribution
            hist = hist / hist.sum(axis=1)[:,None]
            Ym = np.mean(hist, axis=0)
            Ys = np.std(hist, axis=0)
            Ylo, Yhi = np.quantile(hist, ci, axis=0)
            hist_stats[meter] = [Ym, Ys, Ylo, Yhi]
        pickle.dump(hist_stats, open(path, 'wb'))
        return hist_stats


### Multiply duration values by a factor of 2 to get units of eighth notes
def get_onset_histograms(res, parts_data, subdivision, factor=2):
    """
    Tally onset positions within a bar for all parts in a set of pairs.

    Onsets are computed from cumulative note durations and reduced to
    within-bar positions (modulo ``subdivision``) in units of eighth
    notes (``factor=2`` converts quarter-note units).

    Parameters
    ----------
    res : pandas.DataFrame
        Hit table subset filtered to one meter.
    parts_data : dict
        Parts feature dicts keyed by ``part_id``.
    subdivision : int
        Number of grid positions per bar for the given meter.
    factor : int, optional
        Multiplier to convert quarter-note durations to the desired
        grid resolution.  Default is ``2`` (eighth notes).

    Returns
    -------
    numpy.ndarray, shape (n_pairs, subdivision)
        Per-pair onset count at each within-bar position.
    """
    onset_count_all = []
    for q, t in zip(*res[['query', 'target']].values.T):
        onset_count = Counter(round(float(x)*factor, 1) % subdivision for y in [q, t] for x in np.cumsum(parts_data[y][0][0]))
        # No need to normalize by bar, since each position can be found in
        # any bar
        onset_count_all.append(np.array([onset_count.get(i, 0) for i in range(subdivision)]))
    return np.array(onset_count_all)


### Correlation between bar position substitution rate and metrical hierarchy,
### and onset stability
def bar_pos_rate_corr(redo=False):
    """
    Compute R² between within-bar substitution rates and metrical
    hierarchy / onset stability, across PID thresholds.

    For each PID threshold, builds a DataFrame via
    ``load_hierarchy_stability_df`` and computes:

    1. Pearson R² between metrical hierarchy and relative substitution
       rate.
    2. Pearson R² between relative onset stability and relative
       substitution rate.
    3. R² from an OLS regression of substitution rate on (hierarchy,
       end_pos).

    Parameters
    ----------
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    corr : numpy.ndarray, shape (len(pid_list), 3)
        Three R² values for each PID threshold.
    """
    path = PATH_FIG_DATA.joinpath(f"bar_pos_rate_corr.npy")
    if path.exists() and not redo:
        return np.load(path)
    else:
        pid_list = np.arange(0.5, 1, 0.05)
        corr = []
        for ipid in range(pid_list.size):
            df = utils.load_hierarchy_stability_df(ipid)
            corr.append(pearsonr(*df[['hierarchy', 'rel_sub_rate']].values.T)[0]**2)
            corr.append(pearsonr(*df[['rel_stability', 'rel_sub_rate']].values.T)[0]**2)

            # OLS regression of substitution rate on hierarchy + end_pos indicator
            X = sm.add_constant(df[['hierarchy', 'end_pos']].values)
            Y = df['rel_sub_rate'].values
            model = sm.OLS(Y, X)
            results = model.fit()
            corr.append(results.rsquared)
        corr = np.array(corr).reshape(pid_list.size, 3)
        np.save(path, corr)
    return corr
