"""Sequence covariance (Fig. 5): position-position covariance and repetition matrices."""

from collections import Counter

import numpy as np

from thesession.config import METER_LIST, PATH_FIG_DATA
from thesession.alignment import parts as PA
from thesession import utils
from thesession.pipeline.fig1 import get_uniq


###################################################################################################
### Sequence covariance (Fig. 5)

    # Run analyses for Fig 5:
    #    covariance and repetition (separate by mode, dance, mode and dance, all and most common 100 tunes)
    #   don't separate by pid. use all available data to get a strong signal
def data_for_fig5(res0, parts_data, redo=False):
    """
    Compute and save all data needed for Figure 5 (covariance analysis).

    Calls ``part_covariance`` to compute position-position covariance
    matrices for each meter and for the 10 most-represented individual
    tune parts.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    parts_data : dict
        Parts feature dicts.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    None
    """
    print("Computing covariance matrices...")
    part_covariance(res0, parts_data, alpha=0.5, redo=redo,
                    extra_pairs=[(2,4)])


### Get the covariance matrices for sets of tunes grouped by meter,
### and for a few tune families
def part_covariance(res, parts_data, factor0=2, nbars=8, alpha=0.5, redo=False,
                    extra_pairs=None):
    """
    Compute position-position covariance matrices for meter groups and
    the 10 most-represented individual tune parts.

    For each meter, computes the covariance across all pairs with the
    same meter via ``part_covariance_meter``.  Then identifies the top 10
    tune/part combinations by hit count and computes their individual
    covariance matrices.  Any additional ``(tune_id, part_id)`` pairs
    supplied via ``extra_pairs`` are processed after the top-10 set.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table.
    parts_data : dict
        Parts feature dicts.
    factor0 : int, optional
        Grid factor used to convert note durations (default ``2`` for
        eighth-note resolution).
    nbars : int, optional
        Number of bars to include in each sequence window.  Default ``8``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.
    extra_pairs : list of (tune_id, part_id), optional
        Hard-coded ``(tune_id, part_id)`` pairs to process in addition to
        the automatically selected top-10.  Default is ``None``.

    Returns
    -------
    None
    """
    # Average across tunes with the same meter
    for meter in METER_LIST:
        path = PATH_FIG_DATA.joinpath(f"part_cov-{meter.replace('/', '_')}.npy")
        print(f"Running on {meter}")
        part_covariance_meter(res, parts_data, meter, path, alpha=alpha, redo=redo)

    # To find candidate parts that have a lot of variants, first look at individual parts of tunes,
    # sort by most pairs, and pick the first 10
    res_same = res.loc[(res.query_tune==res.target_tune)&(res.query_part==res.target_part)]
    count = Counter(get_uniq(x) for x in res_same[['query', 'target']].values.ravel())
    candidates = sorted(count.items(), key=lambda x: x[1])[::-1]
    part_set = []
    for (tune_id, part_id), num in candidates:
        part_set.append((tune_id, part_id))
        if len(part_set) >= 10:
            break

    # Append any hard-coded extra pairs (skip duplicates)
    if extra_pairs:
        part_set_set = set(part_set)
        for pair in extra_pairs:
            if pair not in part_set_set:
                part_set.append(pair)
                part_set_set.add(pair)

    for tune_id, part_id in part_set:
        # This time round, allow any similar parts from any tune, and any part_id
        idx = (res.query_tune==tune_id)&(res.query_part==part_id)
        print(f"Running on tune {tune_id} part {part_id}. Total: {np.sum(idx)}")
        meter = res.loc[idx, 'target_meter'].iloc[0]
        path = PATH_FIG_DATA.joinpath(f"part_cov-{tune_id}_{part_id}.npy")
        part_covariance_meter(res.loc[idx], parts_data, meter, path, alpha=alpha, redo=redo)


### Calculate position-position covariance matrices
### factor0 is set to 2, so that the positions are only checked at
### eighth note positions
def part_covariance_meter(res, parts_data, meter, path, factor0=2, nbars=8, alpha=0.5, redo=False):
    """
    Compute the position-position covariance and repetition matrices for
    a set of pairs with a given meter.

    For each pair, sequences are placed on a uniform eighth-note grid,
    octave-corrected, and truncated to ``ngrid = nbars * grid_per_bar``
    positions.  The covariance matrix reflects how often positions
    change together, and the repetition matrix identifies positions
    where both sequences share a note but that note differs between the
    two sequences (evidence of preserved repetition under mutation).

    Parameters
    ----------
    res : pandas.DataFrame
        Hit table subset for the desired meter/tune group.
    parts_data : dict
        Parts feature dicts.
    meter : str
        Meter string (e.g. ``'4/4'``).
    path : pathlib.Path
        Cache file path (saved as ``.npy`` with two arrays: ``[cov, rep]``).
    factor0 : int, optional
        Base grid factor (2 = eighth-note resolution).  Default is ``2``.
    nbars : int, optional
        Number of bars per window.  Default is ``8``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute if ``True``.  Default is ``False``.

    Returns
    -------
    cov : numpy.ndarray, shape (ngrid, ngrid)
        Weighted position-position covariance matrix.
    rep : numpy.ndarray, shape (ngrid, ngrid)
        Weighted mean repetition indicator matrix.

    Notes
    -----
    If a pair's grid factor is larger than ``factor0`` (because it
    contains notes shorter than an eighth note), the sequence is
    sub-sampled at every ``ratio``-th position to restore the eighth-note
    grid.  Pairs whose sequence is shorter than ``ngrid`` after gridding
    are skipped.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        grid_per_bar = int(4 * eval(meter) * factor0)
        ngrid = nbars * grid_per_bar

        res = res.loc[(res.target_meter==res.query_meter) & (res.target_meter==meter)
                       & (res.nbars >= nbars)]
        weights = utils.inverse_frequency_weights(res, alpha)
        print(f"Running covariance analysis on {meter} tunes: {len(res)} total")

        rep_mat = []
        eq = []
        idx = []
        for j, i in enumerate(res.index):
            q, t, f, g = res.loc[i, ['query', 'target', 'factor', 'grid_per_bar']]

            # Convert sequences to a standardized grid
            tc1, tc2, factor = PA.part2grid(parts_data[q][0], parts_data[t][0])
            tc1, tc2 = PA.correct_octave_diff(tc1, tc2)

            # If a larger factor was needed due to the presence of notes smaller than
            # eighth notes, throw away any data on positions off the eighth note positions
            if factor > factor0:
                ratio = int(factor / factor0)
                tc1 = tc1[::ratio]
                tc2 = tc2[::ratio]

            # Check that the correct number of grid points are there
            if (tc1.size < ngrid) or (tc2.size < ngrid):
                continue

            # Remove excess
            tc1, tc2 = tc1[:ngrid], tc2[:ngrid]

            # Find the positions that have the same notes in each sequence
            # This indicates where repetition occurs, and where long-range covariance
            # is due to preservation of repetition
            rep_mat.append((tc1[:,None] == tc1[None,:]) & (tc2[:,None] == tc2[None,:]))

            # Get boolean match vector
            b = (tc1 == tc2).astype(float)

            eq.append(b)
            idx.append(j)

        w = weights[idx]
        eq_arr = np.array(eq)
        cov = np.cov(eq_arr.T, aweights=w)

        # Compute "repetition covariance"
        mu = np.average(eq_arr, weights=w, axis=0)
        rep_cov_sum = np.zeros((ngrid, ngrid))
        for i in range(len(idx)):
            dev = eq_arr[i] - mu
            rep_cov_sum += w[i] * rep_mat[i] * np.outer(dev, dev)

        rep_cov = rep_cov_sum / np.sum(w)

        np.save(path, [cov, rep_cov])

        conservation = np.average(eq_arr, weights=w, axis=0)
        conservation_path = path.with_name(path.stem + "_conservation" + path.suffix)

        np.save(conservation_path, conservation)
        return cov, rep_cov
