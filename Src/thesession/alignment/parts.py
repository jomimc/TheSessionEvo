"""
Algorithm for aligning and comparing parts.


"""
from collections import Counter, defaultdict
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

from thesession.config import PATH_CACHE
from thesession import utils


#######################################################
### Aligning parts identified using mmseqs

### Check that total durations add up
def compare_part_duration(part1, part2):
    """
    Check whether two parts have the same total duration.

    Parameters
    ----------
    part1 : tuple
        A (dur, tmidi) tuple where dur is an array of note durations
        (in eighth-note units) and tmidi is an array of absolute MIDI pitches.
    part2 : tuple
        A (dur, tmidi) tuple with the same structure as part1.

    Returns
    -------
    bool
        True if the summed durations of both parts are equal, False otherwise.
    """
    return np.sum(part1[0]) == np.sum(part2[0])


### Compare two parts by comparing pitches aligned on a grid
def compare_parts(part1, part2):
    """
    Compare two parts by placing their pitch-class sequences on a shared
    time grid and computing the fraction of grid positions that match.

    Parameters
    ----------
    part1 : tuple
        A (dur, tmidi) tuple for the first part.
    part2 : tuple
        A (dur, tmidi) tuple for the second part.

    Returns
    -------
    equal_dur : bool
        True if both parts have the same total duration.
    frac_match : float
        Fraction of grid positions where the pitch classes agree.
        Returns np.nan if durations differ or if no common denominator
        can be found.

    Notes
    -----
    Pitch classes (tchroma = tmidi % 12) are compared after expanding
    each note to fill its grid-quantised duration slots. The grid
    resolution is determined by the smallest common integer factor of
    all distinct duration values across both parts.
    """
    equal_dur = compare_part_duration(part1, part2)
    if not equal_dur:
        return False, np.nan

    # Put tunes on a grid
    factor = utils.get_common_denominator([part1[0], part2[0]])
    if factor == 0:
        print("Common denominator not found!")
        return True, np.nan

    tc1 = utils.get_tchroma_grid(part1[1], part1[0], factor)
    tc2 = utils.get_tchroma_grid(part2[1], part2[0], factor)

    frac_match = np.mean(tc1 == tc2)

    return True, frac_match


### Convert parts to a sequence of pitches on a regular grid
def part2grid(part1, part2, factor=None):
    """
    Convert two parts to pitch-class grid sequences of equal resolution.

    Parameters
    ----------
    part1 : tuple
        A (dur, tmidi) tuple for the first part.
    part2 : tuple
        A (dur, tmidi) tuple for the second part.
    factor : int or None, optional
        Pre-computed grid factor (smallest integer such that all
        durations * factor are integers). If None, it is computed
        automatically from the union of both parts' duration arrays.

    Returns
    -------
    tc1 : numpy.ndarray
        Pitch-class grid sequence for part1 (length = total_dur * factor).
    tc2 : numpy.ndarray
        Pitch-class grid sequence for part2.
    factor : int
        The grid factor that was used (returned so callers can cache it).
    """
    if isinstance(factor, type(None)):
        factor = utils.get_common_denominator([part1[0], part2[0]])
    tc1 = utils.get_tchroma_grid(part1[1], part1[0], factor)
    tc2 = utils.get_tchroma_grid(part2[1], part2[0], factor)
    return tc1, tc2, factor


### Check if parts differ by an octave,
### and if so correct by changing the pitch of one part
def correct_octave_diff(tc1, tc2, threshold=8):
    """
    Correct a systematic octave transposition between two pitch sequences.

    If the mean pitch of tc1 and tc2 differ by more than `threshold`
    semitones, tc1 is shifted by the nearest multiple of 12 semitones
    to bring the two sequences into the same octave register.

    Parameters
    ----------
    tc1 : numpy.ndarray
        Grid-expanded pitch (or pitch-class) sequence for the first part.
    tc2 : numpy.ndarray
        Grid-expanded pitch (or pitch-class) sequence for the second part.
    threshold : int, optional
        Minimum mean-pitch difference (in semitones) that triggers a
        correction. Default is 8.

    Returns
    -------
    tc1 : numpy.ndarray
        Octave-corrected version of tc1 (tc2 is never modified).
    tc2 : numpy.ndarray
        Unchanged tc2.

    Notes
    -----
    Only tc1 is adjusted; tc2 is returned as-is. The shift applied is
    ``-12 * round(diff / 12)``, which is the closest multiple of an
    octave that minimises the residual mean-pitch difference.
    """
    m1 = np.mean(tc1)
    m2 = np.mean(tc2)
    diff = m1 - m2
    if abs(diff) > threshold:
        # Shift tc1 by the nearest integer number of octaves to align registers
        return tc1 - 12 * np.round(diff / 12), tc2
    return tc1, tc2


### Align and compare two parts
def analyse_part_alignment(part1, part2, meter):
    """
    Align two parts on a time grid and collect detailed per-position
    substitution and match statistics.

    Parameters
    ----------
    part1 : tuple
        A (dur, tmidi) tuple for the first part.
    part2 : tuple
        A (dur, tmidi) tuple for the second part.
    meter : str
        Time-signature string in Python-evaluable fraction notation
        (e.g. ``'6/8'``, ``'4/4'``). Passed to ``eval()`` to obtain the
        numeric meter value used to compute bar boundaries.

    Returns
    -------
    out : dict
        Dictionary with the following keys:

        sub_bar : numpy.ndarray of int
            Bar index (0-based) of each mismatching grid position.
        sub_pos : numpy.ndarray of float
            Within-bar position (0.0–1.0) of each mismatch, expressed as
            a fraction of the bar length.
        sub_notes : numpy.ndarray, shape (n_mismatches, 2)
            Pairs of (tc1_pitch, tc2_pitch) at each mismatching position.
        sub_dist : numpy.ndarray of int
            Absolute pitch difference at each mismatching grid position.
        match_notes : collections.Counter
            Counts of each pitch value at positions where both parts agree.
        grid_per_bar : int
            Number of grid positions per bar (= 4 * eval(meter) * factor).
        factor : int
            The grid factor used to quantise durations.

    Notes
    -----
    The grid factor is computed jointly from both parts so that all note
    durations map to integer grid slots. Bar boundaries are computed as
    ``onset // grid_per_bar``; within-bar position as
    ``(onset % grid_per_bar) / grid_per_bar``.

    .. warning::
        The ``grid_per_bar`` formula assumes the factor has already
        accounted for the eighth-note unit convention; verify if the
        meter encoding changes.
    """
    # Put tunes on a grid
    tc1, tc2, factor = part2grid(part1, part2)

    # Correct for potential octave difference
    tc1, tc2 = correct_octave_diff(tc1, tc2)

    # Check this number! It might be off by a factor of two (or else I have already corrected it!)
    grid_per_bar = int(4 * eval(meter) * factor)

    # Boolean mask: True wherever the two sequences disagree
    idx = tc1 != tc2

    # Assign every grid position an absolute onset index
    onset = np.arange(len(tc1))
    # Position within the bar (grid units)
    bar_onset = onset % grid_per_bar
    # Which bar each grid position belongs to
    bar = onset // grid_per_bar

    out = {'sub_bar': bar[idx],
           # Normalise to [0, 1) so positions are meter-independent
           'sub_pos': bar_onset[idx] / grid_per_bar,
           # Stack both pitch sequences and take only the mismatching columns
           'sub_notes': np.array([tc1, tc2])[:,idx].T,
           'sub_dist': np.abs(tc1[idx] - tc2[idx]),
           'match_notes': Counter(tc1[~idx]),
           'grid_per_bar':grid_per_bar,
           'factor':factor
           }

    return out


### Look for identical parts and cluster them,
### then prune hits pertaining to all but one per group
def prune_identical_parts(res, parts):
    """
    Remove redundant hits caused by duplicate (identical) parts.

    When multiple settings of the same tune are byte-for-byte identical,
    MMseqs2 will report cross-hits among all pairs. This function detects
    groups of mutually identical parts, selects one canonical
    representative per group (the lexicographically smallest part_id),
    and drops all rows of ``res`` that involve any non-representative
    duplicate.

    Parameters
    ----------
    res : pandas.DataFrame
        MMseqs2 result table with at least the columns ``query``,
        ``target``, ``fident``, and ``alnlen``. Rows are pairwise
        alignment hits between parts.
    parts : dict
        Mapping from part_id (str) to ``(part, nbars)`` where
        ``part = (dur_array, tmidi_array)``.

    Returns
    -------
    pandas.DataFrame
        A filtered copy of ``res`` containing only rows where neither
        ``query`` nor ``target`` is a redundant duplicate.

    Notes
    -----
    Identity is defined strictly: ``fident == 1.0`` *and* the alignment
    length equals the original sequence length for *both* the query and
    the target. The second condition rules out cases where a shorter
    sequence is fully contained in a longer one but the longer one has
    additional notes.

    Connected components of the resulting identity graph (one node per
    part_id, one edge per identical pair) are computed with NetworkX.
    The representative chosen for each component is deterministic:
    ``sorted(component)[0]``, i.e. the alphabetically first part_id.
    """
    # Create the graph of identical parts
    G = nx.Graph()

    # Get part lengths
    uniq_parts = np.unique(np.concatenate([res['query'].unique(), res['target'].unique()]))
    # Number of notes (not grid positions) in each part — used to validate alnlen
    part_length = {p: len(parts[p][0][0]) for p in uniq_parts}
    res['qlen'] = res['query'].map(part_length)
    res['tlen'] = res['target'].map(part_length)

    # Identical tunes must not only have "fident" ("fraction identity") = 1,
    # but the alignment length must also be the same size as both original sequences
    same_len = (res.alnlen==res['qlen']) & (res.alnlen==res['tlen'])
    identical_parts = res.loc[(res.fident==1)&(same_len), ['query', 'target']].values
    # Each edge connects two parts that are truly identical
    G.add_edges_from(map(tuple, identical_parts))

    # Get connected components (groups of identical parts)
    components = list(nx.connected_components(G))

    # Choose deterministic representative per component, and note the others
    reps = []
    to_remove = []
    for comp in components:
        sort_comp = sorted(comp)
        reps.append(sort_comp[0])  # deterministic (smallest query setting id)
        to_remove.extend(sort_comp[1:])

    # Keep only rows where neither endpoint is a discarded duplicate
    return res.loc[(~res["query"].isin(to_remove))&(~res["target"].isin(to_remove))]


### Filter and annotate pairs of parts identified using mmseqs
def annotate_res(df, df_parts, res, parts, redo=False):
    """
    Filter, annotate, and cache the full MMseqs2 hit table.

    Starting from the raw MMseqs2 output ``res``, this function:

    1. Removes self-hits and redundant duplicate-part hits.
    2. Joins each hit with setting-level metadata (tune ID, part number,
       meter, dance type, mode).
    3. Rejects hits where the two parts have different total durations or
       different meters, or where the pitch-class agreement falls outside
       (0.5, 1.0) — the range of "true but non-identical variants".
    4. Validates that the ABC meter annotation is consistent with the
       actual note content.
    5. Runs ``analyse_part_alignment`` on every surviving hit to produce
       per-pair substitution statistics.
    6. Pickles all three result tables to ``PATH_CACHE`` so subsequent
       calls can skip recomputation unless ``redo=True``.

    Parameters
    ----------
    df : pandas.DataFrame
        Setting-level metadata table with columns ``setting_id``,
        ``meter``, and ``type`` (dance type).
    df_parts : pandas.DataFrame
        Part-level metadata table with columns ``part_id``,
        ``setting_id``, ``tune_id``, ``part_no``, and ``num_parts``.
    res : pandas.DataFrame
        Raw MMseqs2 result table with columns ``query``, ``target``,
        ``fident``, and ``alnlen``.
    parts : dict
        Mapping from part_id (str) to ``(part, nbars)`` where
        ``part = (dur_array, tmidi_array)``.
    redo : bool, optional
        If False (default) and cached results exist, load and return
        them without recomputation. Set to True to force recomputation.

    Returns
    -------
    res : pandas.DataFrame
        Fully annotated hit table (all hits that survived the
        duplicate-pruning and self-hit removal steps).
    res0 : pandas.DataFrame
        Filtered subset of ``res`` containing only true variant pairs:
        equal duration, equal meter, 0.5 < frac_eq < 1.0, and
        consistent ABC meter annotation. Augmented with columns
        ``total_dur``, ``factor``, ``grid_per_bar``, ``nbars``,
        ``correct_meter``, ``*_mode``, and ``*_tunecount``.
    mismatches : pandas.DataFrame
        One row per row of ``res0``, each row being the dict returned by
        ``analyse_part_alignment`` (columns: ``sub_bar``, ``sub_pos``,
        ``sub_notes``, ``sub_dist``, ``match_notes``, ``grid_per_bar``,
        ``factor``).

    Notes
    -----
    The ``frac_eq`` upper bound of 1.0 deliberately excludes identical
    pairs; those are handled separately by ``prune_identical_parts``.
    The lower bound of 0.5 is a heuristic to exclude spurious MMseqs2
    hits that happen to share a short high-identity sub-region.

    Caching uses ``pd.DataFrame.to_pickle`` / ``pd.read_pickle``; the
    three cache files are located under ``PATH_CACHE`` (defined in
    ``thesession.config``).
    """
    path_results = [PATH_CACHE.joinpath(n) for n in ["pairs_thesession_parts.pkl",
                                                     "pairs_thesession_parts_hits.pkl",
                                                     "pairs_thesession_parts_mismatches.pkl"]]
    if np.all([p.exists() for p in path_results]) and not redo:
        return [pd.read_pickle(p) for p in path_results]

    # Remove self-hits
    res = res.loc[res['query'] != res['target']]

    # Remove hits from redundant parts
    # i.e. ensure that for all groups of identical parts,
    # hits will only show up for one of them

    # Unpack name / identifiers
    # Build a nested dict: multikey[column_name][part_id] = value
    # so we can later vectorise the mapping onto the res DataFrame
    cols = ['setting_id', 'tune_id', 'part_no', 'num_parts']
    multikey = defaultdict(dict)
    for p, vals in zip(df_parts.part_id, df_parts[cols].values):
        for c, v in zip(cols, vals):
            multikey[c][p] = v

    # Attach metadata columns for both query and target sides of each hit
    col2 = ['setting', 'tune', 'part', 'num_parts']
    for a in ['query', 'target']:
        for c1, c2 in zip(cols, col2):
            c3 = f"{a}_{c2}"
            res[c3] = res[a].map(multikey[c1])

    # Annotate in_fam
    # True when query and target belong to the same tune (within-family comparison)
    res['in_fam'] = res['target_tune'] == res['query_tune']


    # Annotate meter and dance
    meter_key = {s:m for s, m in zip(df['setting_id'], df['meter'])}
    res['target_meter'] = res['target_setting'].map(meter_key)
    res['query_meter'] = res['query_setting'].map(meter_key)
    res['eq_meter'] = res['target_meter'] == res['query_meter']

    dance_key = {s:m for s, m in zip(df['setting_id'], df['type'])}
    res['target_dance'] = res['target_setting'].map(dance_key)
    res['query_dance'] = res['query_setting'].map(dance_key)
    res['eq_dance'] = res['target_dance'] == res['query_dance']

    # Align and compare matches
    # compare_parts returns (equal_dur, frac_match) for every hit
    out = np.array([compare_parts(parts[i][0], parts[j][0]) for i, j in zip(res['query'], res['target'])])
    res['eq_dur'] = out[:,0].astype(bool)
    res['frac_eq'] = out[:,1].astype(float)

    # Reduce to true hits
    # Require: equal duration, equal meter, and pitch-class agreement
    # strictly between 50% and 100% (identical pairs are excluded)
    res0 = res.loc[(res.eq_dur)&(res.eq_meter)&(res.frac_eq>0.5)&(res.frac_eq<1)]

    # Annotate duration and discretization details
    res0['total_dur'] = res0['query'].apply(lambda x: np.sum(parts[x][0][0])) # Total duration in eigth note units
    res0['factor'] = [utils.get_common_denominator([parts[q][0][0], parts[t][0][0]])
                      for q, t in zip(res0['query'], res0['target'])] # Factor used to map lowest duration to grid spacing
    res0["grid_per_bar"] = [int(4 * eval(m) * f) for m, f in zip(res0.target_meter, res0.factor)] # Number of grid points per bar
    res0['nbars'] = res0['query'].apply(lambda q: parts[q][1]) # Number of bars in part

    # Remove songs where meter in ABC is not the same as in the header
    # The expected total duration per bar is 4 * meter eighth-note units;
    # disagreement indicates an inconsistency in the ABC notation
    res0 = res0.loc[(res0['total_dur'] / res0['nbars']) == (4 * res0.target_meter.apply(eval))]

    # Check that the meter is correct (sometimes header annotation is wrong)
    res0['correct_meter'] = (res0['total_dur'] / res0['nbars']) == (4 * res0.target_meter.apply(eval))

    # Annotate mode
    # Use pitch-class histogram of tmidi to infer major/minor/mixolydian/dorian
    res0['target_mode'] = [utils.check_mode(parts[t][0][1] % 12) for t in res0['target']]
    res0['query_mode'] = [utils.check_mode(parts[t][0][1] % 12) for t in res0['query']]

    # Annotate tune counts
    # Count how many times each tune appears across all hit pairs (query + target side)
    # to later compute inverse-frequency weights that down-weight prolific tunes
    tune_counts = Counter(res[['query_tune','target_tune']].values.ravel())
    res0['query_tunecount'] = res0['query_tune'].map(tune_counts)
    res0['target_tunecount'] = res0['target_tune'].map(tune_counts)

    # Align and compare true hits
    mismatches = pd.DataFrame([analyse_part_alignment(parts[q][0], parts[t][0], m)
                               for q, t, m in zip(res0['query'], res0['target'], res0['target_meter'])])

    # Save results
    for p, r in zip(path_results, [res, res0, mismatches]):
        r.to_pickle(p)

    return res, res0, mismatches


### Since I have mapped notes to a grid, this introduces a new
### degree of freedom, since I need to avoid artefacts due to overcounting
### that might occur when a tune has greater number of grids per bar...
### So I will normalize observations by reweighting them compared to eighth notes
def subs_to_observations(res, mismatches, alpha=0.5):
    """
    Aggregate pairwise substitution and match counts into a
    pitch-class transition observation table, correcting for grid
    resolution and tune frequency.

    Each substitution or match contributes a weight proportional to its
    duration in units of *one bar* (not raw grid slots), so that tunes
    with finer grid resolution do not dominate the counts. An additional
    inverse-frequency weight down-weights tunes that appear in many
    pairs.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table (``res0`` from ``annotate_res``) with columns
        ``target_meter``, ``grid_per_bar``, ``query_tunecount``, and
        ``target_tunecount``.
    mismatches : pandas.DataFrame
        Per-pair alignment details produced by ``analyse_part_alignment``,
        with columns ``sub_notes`` (array of (pitch_a, pitch_b) pairs) and
        ``match_notes`` (Counter of matched pitches).
    alpha : float, optional
        Exponent for the inverse-frequency weight: weight = (geometric
        mean tune count)^(-alpha). Default is 0.5 (square-root
        down-weighting).

    Returns
    -------
    obs : collections.defaultdict of float
        Mapping from (pitch_class_a, pitch_class_b) → accumulated
        observation weight. Diagonal entries (a == b) correspond to
        conserved positions; off-diagonal entries are substitutions.
        Pitch classes are integers in [0, 11] (tchroma = tmidi % 12).

    Notes
    -----
    The normalisation unit for a single grid position in a pair is::

        unit = eval(meter) / grid_per_bar * weight

    Since ``grid_per_bar = 4 * eval(meter) * factor``, this simplifies
    to ``1 / (4 * factor)``, which is the fraction of a bar that one
    grid slot occupies — equivalent to expressing durations in bars
    regardless of the grid resolution. This ensures that a note lasting
    one eighth-note unit counts the same whether the pair's grid has
    ``factor=1`` or ``factor=4``.
    """
    obs = defaultdict(float)
    weights = utils.inverse_frequency_weights(res, alpha)
    for subs, m, g, w in zip(mismatches.sub_notes, res.target_meter, res.grid_per_bar, weights):
        # unit: contribution of one grid position, in bar-fraction units, after frequency weighting
        unit = eval(m) / g * w
        for a, b in zip(*subs.T):
            # Reduce to pitch class before accumulating, so octave
            # differences from correct_octave_diff do not create spurious entries
            obs[(a%12,b%12)] += unit

    for matches, m, g, w in zip(mismatches.match_notes, res.target_meter, res.grid_per_bar, weights):
        unit = eval(m) / g * w
        for k, v in matches.items():
            # Matched notes contribute v grid slots at the diagonal entry (k, k)
            obs[(k%12,k%12)] += unit * v

    return obs


### Get the melodic interval distribution from tunes
def get_mint_dist(tunes):
    """
    Compute the normalised melodic interval distribution over a
    collection of tunes.

    Parameters
    ----------
    tunes : dict
        Mapping from tune identifier to a dict containing at least a
        ``'tmidi'`` key holding an array of absolute MIDI pitches for
        that tune.

    Returns
    -------
    X : numpy.ndarray of int
        Interval sizes in semitones, ranging from 1 to 17 inclusive.
    Y : numpy.ndarray of float
        Normalised count for each interval size (sums to 1.0).

    Notes
    -----
    Only ascending or descending steps (absolute interval) are tallied;
    unison (0 semitones) and intervals larger than 17 semitones are
    ignored. The distribution is computed by summing absolute first
    differences of the MIDI pitch sequence across all tunes.
    """
    X = np.arange(1, 18)
    tune_dists = []
    for d in tunes.values():
        C = Counter(np.abs(np.diff(d['tmidi'])))
        counts = np.array([C.get(x, 0) for x in X], dtype=float)
        total = counts.sum()
        if total > 0:
            tune_dists.append(counts / total)
    Y = np.mean(tune_dists, axis=0)
    return X, Y


#######################################################
### Multiple part alignment


def get_msa_family(res, parts, tune_id, p=0, min_pid=0.85, max_grid=16, nbars=8, factor=2, part_list=None):
    """
    Build a multiple-sequence alignment (MSA) matrix for all settings
    of a given tune part that meet a minimum pairwise identity threshold.

    The function collects every part_id that appears in a within-family
    hit pair for the requested tune and part number, places each on a
    fixed-length grid, corrects octave differences relative to the first
    part, and stacks the resulting sequences into a 2-D array.

    Parameters
    ----------
    res : pandas.DataFrame
        Annotated hit table (``res0`` from ``annotate_res``) with
        columns ``query_tune``, ``target_tune``, ``fident``,
        ``target_meter``, ``query_meter``, ``query_part``,
        ``target_part``, ``query``, and ``target``.
    parts : dict
        Mapping from part_id (str) to ``(part, nbars)`` where
        ``part = (dur_array, tmidi_array)``.
    tune_id : int or str
        The tune identifier whose settings are to be aligned.
    p : int, optional
        Part index within each setting (0 = first part). Default is 0.
    min_pid : float, optional
        Minimum MMseqs2 fractional identity (``fident``) required for a
        hit to be included. Default is 0.85.
    max_grid : int, optional
        Maximum number of grid positions per bar. Together with
        ``nbars``, this sets the fixed MSA length
        ``ngrid = nbars * max_grid``. Default is 16.
    nbars : int, optional
        Number of bars to retain from each sequence. Default is 8.
    factor : int, optional
        Grid factor passed directly to ``part2grid``; all parts are
        placed on the same grid regardless of their individual duration
        sets. Default is 2.
    part_list : array-like or None, optional
        If provided, skip the automatic hit-table query and use this
        explicit list of part_ids. Default is None (auto-detect).

    Returns
    -------
    part_list : numpy.ndarray of str
        The part_ids (in the same order as the rows of ``msa``) that
        were successfully included. Returns an empty list if no eligible
        parts were found.
    msa : numpy.ndarray, shape (n_parts, ngrid)
        Each row is the grid-expanded pitch sequence for one setting,
        truncated to ``ngrid`` positions. Returns an empty list if no
        eligible parts were found.

    Notes
    -----
    The first two parts in ``part_list`` are always used as the seed
    pair (they anchor the octave correction). Subsequent parts are each
    independently octave-corrected relative to the first part and
    included only if their grid sequence is at least ``ngrid`` positions
    long (i.e. the part is not shorter than the requested window).

    The fixed grid length ``ngrid = nbars * max_grid`` is a design
    choice: using a uniform window avoids ragged arrays and ensures that
    positional statistics (e.g. conservation per column) are comparable
    across families. Parts that are shorter than ``ngrid`` after
    gridding are silently excluded via the ``tc3.size >= ngrid`` guard.
    """
    if isinstance(part_list, type(None)):
        # Collect all unique part_ids that appear in within-family hits for this tune/part
        part_list = np.unique(res.loc[(res['query_tune']==tune_id)&(res['target_tune']==tune_id)
                                      &(res.fident>=min_pid)&(res.target_meter==res.query_meter)
                                      &(res['query_part']==p)&(res['target_part']==p),
                                      ['query', 'target']].values)

    if len(part_list) == 0:
        return [], []

    # Fixed MSA column count: each bar contributes max_grid positions
    ngrid = nbars * max_grid

    # Seed the MSA with the first two parts, applying octave correction
    tc1, tc2 = part2grid(parts[part_list[0]][0], parts[part_list[1]][0], factor)[:2]
    tc1, tc2 = correct_octave_diff(tc1, tc2)
    tc_list = [tc1[:ngrid], tc2[:ngrid]]
    idx_out = [0,1]
    for i in range(2, len(part_list)):
        # Always align each additional part against part_list[0] for a consistent octave reference
        tc1, tc3 = part2grid(parts[part_list[0]][0], parts[part_list[i]][0], factor)[:2]
        tc1, tc3 = correct_octave_diff(tc1, tc3)
        if tc3.size >= ngrid:
            tc_list.append(tc3[:ngrid])
            idx_out.append(i)
    return part_list[idx_out], np.array(tc_list)


def get_position_conservation(msa):
    """
    Compute per-column Shannon entropy of a multiple-sequence alignment.

    Parameters
    ----------
    msa : numpy.ndarray, shape (n_sequences, n_positions)
        Integer pitch (or pitch-class) MSA matrix as returned by
        ``get_msa_family``.

    Returns
    -------
    H : numpy.ndarray of float, shape (n_positions,)
        Shannon entropy at each alignment column. Columns where all
        sequences carry the same value have H = 0 (fully conserved).

    Notes
    -----
    Entropy is computed from the normalised frequency of each distinct
    value in the column using ``scipy.stats.entropy``. A fully conserved
    column (all sequences identical) short-circuits to 0 without calling
    ``entropy``, avoiding a log(0) edge case.
    """
    N = msa.shape[1]
    H = np.zeros(N, float)
    for i in range(N):
        count = np.array(list(Counter(msa[:,i]).values()))
        if count.size == 1:
            # All sequences carry the same value: entropy is exactly zero
            H[i] = 0
        else:
            H[i] = entropy(count / count.sum())
    return H


def count_ngrams(parts, part_list, factor=2, context=4):
    """
    Count all pitch-class n-grams of a given length across a set of parts.

    Parameters
    ----------
    parts : dict
        Mapping from part_id (str) to ``(part, nbars)`` where
        ``part = (dur_array, tmidi_array)``.
    part_list : array-like of str
        Subset of part_ids to process.
    factor : int, optional
        Grid factor used to expand note durations into grid slots.
        Default is 2.
    context : int, optional
        N-gram length (number of consecutive pitch-class grid positions).
        Default is 4.

    Returns
    -------
    count : numpy.ndarray, shape (12,) * context
        A ``context``-dimensional array indexed by pitch class (0–11)
        where ``count[c0, c1, ..., c_{n-1}]`` is the total number of
        times that n-gram appears across all parts in ``part_list``.

    Notes
    -----
    Parts that contain any note shorter than ``1/factor`` (i.e. whose
    duration would be rounded to zero grid slots) are skipped entirely
    to avoid artefacts from the integer quantisation.
    """
    count = np.zeros([12]*context, float)
    for p in tqdm(part_list):
        dur, midi = parts[p][0]
        # Skip parts whose shortest note cannot be represented on the grid
        if np.any(dur < 1 / factor):
            continue
        tc = np.array(utils.get_tchroma_grid(midi, dur, factor) % 12, int)
        for i in range(len(tc) - context + 1):
            count[tuple(tc[i:i+context])] += 1
    return count


def get_novelty_profile(res, parts, q, N=4):
    """
    Compute a novelty profile for a query part against a background
    n-gram frequency table.

    For each position in the MSA of the query part's tune family, the
    function looks up how often the N-gram starting at that position
    appears in the background corpus (all parts *not* related to the
    query), giving a measure of how common or novel each melodic phrase
    is.

    Parameters
    ----------
    res : pandas.DataFrame
        Annotated hit table with columns ``query`` and ``target``.
    parts : dict
        Mapping from part_id (str) to ``(part, nbars)`` where
        ``part = (dur_array, tmidi_array)``.
    q : str
        The query part_id whose novelty profile is to be computed.
    N : int, optional
        N-gram length used for the frequency lookup. Default is 4.

    Returns
    -------
    msa_prev : numpy.ndarray, shape (n_sequences, n_positions - N + 1)
        For each sequence in the family MSA and each starting position,
        the background corpus count of the N-gram of length ``N``
        beginning at that position. Lower values indicate rarer
        (more novel) phrases.

    Notes
    -----
    The background corpus excludes all parts that appear in any hit
    involving the query (both as query and as target), ensuring that the
    novelty score is not inflated by the query's own family variants.

    N-gram counts are computed at grid factor 2 with context 4 via
    ``count_ngrams``. The MSA is obtained via ``get_msa`` (not defined
    in this module) with a minimum identity threshold of 0.5 and 8 bars.
    """
    # Collect all parts related to the query (its family) to exclude from background
    test = np.unique(res.loc[(res['query']==q)|(res['target']==q), ['query', 'target']].values)
    parts_keys = np.array(list(parts.keys()))
    # Background = everything outside the query's hit neighbourhood
    not_in_test = np.array(list(set(parts_keys).difference(test)))
    count = count_ngrams(parts, not_in_test, 2, 4)

    msa = get_msa(res, parts, q, 0.5, 8)
    # Reduce to pitch classes for n-gram indexing into the count array
    msa_tc = np.array(msa % 12, int)
    # For each sequence and each position, look up the background frequency
    # of the N-gram starting at that position
    msa_prev = np.array([[count[tuple(msa_tc[i,j:j+N])] for j in range(msa.shape[1] - N + 1)] for i in range(msa.shape[0])])
    return msa_prev
