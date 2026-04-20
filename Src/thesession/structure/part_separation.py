"""
Algorithm for separating tunes into parts including evaluationg code.


"""
from collections import Counter, defaultdict
import json
import pickle
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesession.io import tune_loader as load_tunes
from thesession.io import tune_parser as TP
from thesession.config import *
from thesession import utils


#######################################################
### Load bars

### Separate a setting into its bars
def extract_bars(tunes):
    """
    Slice a setting's flat note arrays into per-bar lists using bar boundary indices.

    Parameters
    ----------
    tunes : dict
        A setting dict containing at minimum the keys:
        ``'bars'`` (array of note indices marking bar start positions),
        ``'dur'`` (array of note durations in eighth-note units), and
        ``'tmidi'`` (array of absolute MIDI pitch values).

    Returns
    -------
    bar_total_dur : numpy.ndarray
        1-D array of shape ``(nbars,)`` giving the total duration of each bar
        (sum of all note durations within that bar).
    bar_dur : list of numpy.ndarray
        List of length ``nbars``; each element is the sub-array of note
        durations belonging to that bar.
    bar_tmidi : list of numpy.ndarray
        List of length ``nbars``; each element is the sub-array of MIDI
        pitches belonging to that bar.
    """
    bars, dur, tmidi = [tunes[x] for x in ['bars', 'dur', 'tmidi']]
    bar_dur, bar_tmidi = [], []
    # Slice between consecutive bar boundary indices to isolate each bar's notes
    for i in range(len(bars) - 1):
        bar_dur.append(dur[bars[i]:bars[i+1]])
        bar_tmidi.append(tmidi[bars[i]:bars[i+1]])

    # The last bar runs from the final boundary index to the end of the array
    bar_dur.append(dur[bars[-1]:])
    bar_tmidi.append(tmidi[bars[-1]:])

    bar_total_dur = [np.sum(d) for d in bar_dur]

    return np.array(bar_total_dur), bar_dur, bar_tmidi


### Remove pickups and check that durations add up
def screen_bars(tunes, bar_len=None, min_bars=8):
    """
    Validate a setting's bars, strip any pickup bar, and confirm bar-count
    divisibility.

    A setting passes screening if:

    * all bars except (optionally) the first have the expected total duration,
    * the remaining bar count is a multiple of ``min_bars``.

    Parameters
    ----------
    tunes : dict
        A setting dict as described in ``extract_bars``.
    bar_len : float or None, optional
        Expected total duration of a full bar in eighth-note units.  If
        ``None`` (default), the maximum observed bar duration is used as the
        reference length.
    min_bars : int, optional
        Part length in bars; bar count after pickup removal must be a multiple
        of this value.  Default is ``8`` (standard for jigs/reels).

    Returns
    -------
    use_okay : bool
        ``True`` if the setting passes all checks, ``False`` otherwise.
    bar_dur : list of numpy.ndarray
        Per-bar duration sub-arrays after pickup removal (empty list on
        failure).
    bar_tmidi : list of numpy.ndarray
        Per-bar MIDI pitch sub-arrays after pickup removal (empty list on
        failure).

    Notes
    -----
    A short first bar is treated as a pickup (anacrusis) and silently
    dropped.  Any other short bar indicates a transcription error and causes
    the entire setting to be rejected.
    """
    bar_total_dur, bar_dur, bar_tmidi = extract_bars(tunes)

    # Default to the longest bar as the reference full-bar duration
    if isinstance(bar_len, type(None)):
        bar_len = np.max(bar_total_dur)

    # Find bars that are not long enough
    idx = np.where(np.array(bar_total_dur) < bar_len)[0]

    # Check for bars with missing beats (apart from pickup)
    # idx != 0 isolates any short bar that is *not* the first bar; these are
    # genuine transcription errors rather than pickups and cannot be recovered
    if np.sum(idx != 0) > 0:
        return False, [], []

    # Check for pickup
    # If only bar 0 is short, it is a pickup; discard it before proceeding
    if 0 in idx:
        bar_tmidi = bar_tmidi[1:]
        bar_dur = bar_dur[1:]

    # Check for correct length of tune
    # After pickup removal the remaining bars must fill an integer number of parts
    if (len(bar_tmidi) % min_bars) != 0:
        return False, [], []

    return True, bar_dur, bar_tmidi


#######################################################
### Separate tunes into parts


### This algorithm starts by separating parts into 8-bar segments,
### and then merges neighboring segments if they are very similar.
### The output is a list of parts.
### For most tunes, min_bars = 8. For slides, this should techically be 4.
def separate_tune_into_parts(tmidi, dur, cutoff=0.8, min_bars=8):
    """
    Split a tune's bar arrays into structural parts (A part, B part, etc.),
    merging adjacent repeated sections that exceed a chroma-similarity
    threshold.

    The algorithm proceeds in two stages:

    1. **Chunking** — the bars are grouped into non-overlapping ``min_bars``-bar
       segments (typically 8 bars each), called *initial parts*.
    2. **Merging** — adjacent initial parts are compared using a time-grid
       chroma representation.  If their note-by-note chroma agreement exceeds
       ``cutoff`` they are treated as a repeated performance of the same
       structural part and concatenated into a single 16-bar entry.

    Merging is greedy and left-to-right: once two adjacent parts are merged,
    neither can participate in a further merge with its remaining neighbour.

    Parameters
    ----------
    tmidi : list of numpy.ndarray
        Per-bar arrays of absolute MIDI pitch values, one array per bar.
        Length must be a multiple of 8.
    dur : list of numpy.ndarray
        Per-bar arrays of note durations in eighth-note units, parallel to
        ``tmidi``.
    cutoff : float, optional
        Minimum chroma-grid agreement fraction (0–1) required to merge two
        adjacent initial parts.  Default is ``0.8``.
    min_bars : int, optional
        Number of bars per initial chunk.  Default is ``8``.

    Returns
    -------
    parts_final : list of tuple
        Each element is ``(dur_array, tmidi_array)`` for one structural part,
        where the arrays are the concatenated note-level data for all bars in
        that part.
    part_nbars : numpy.ndarray
        1-D integer array whose *k*-th entry gives the number of bars in
        ``parts_final[k]`` (8 for an unmerged chunk, 16 for a merged pair).

    Notes
    -----
    Chroma similarity is computed over a time-quantised grid so that note
    duration is taken into account: each grid cell corresponds to one
    quantum unit (the smallest common denominator of all note durations in
    both parts), and the grid value is the chroma class of the note sounding
    at that time point.  The fraction of grid cells where both parts share
    the same chroma class is the similarity score.

    For slides, ``min_bars`` should be set to ``4`` rather than the default
    ``8``.
    """
    # Join bars into parts composed of "min_bars" bars each
    # Each initial part is a (dur_concat, tmidi_concat) tuple spanning 8 bars
    nparts = len(tmidi) // 8
    parts_init = [(np.concatenate(dur[i*8:(i+1)*8]),
                   np.concatenate(tmidi[i*8:(i+1)*8])) for i in range(nparts)]

    # Calculate similarity between neighboring parts
    # We compare each adjacent pair to identify likely AABB-style repetitions
    part_similarity = np.zeros(nparts - 1)
    for i in range(nparts - 1):
        # Find the smallest time quantum that aligns both parts' duration grids;
        # this normalises across different note subdivisions (triplets, etc.)
        factor = utils.get_common_denominator([parts_init[i][0], parts_init[i+1][0]])
        if factor == 0:
            print(set(np.concatenate([parts_init[i][0], parts_init[i+1][0]])))
            raise Exception("Common denominator not found!")

        # Build time-expanded chroma grids: each note occupies (dur * factor)
        # consecutive cells, so the arrays have commensurate length for comparison
        tc_grid1 = utils.get_tchroma_grid(parts_init[i][1], parts_init[i][0], factor)
        tc_grid2 = utils.get_tchroma_grid(parts_init[i+1][1], parts_init[i+1][0], factor)
        # Fraction of time-grid positions where both parts share the same chroma class
        part_similarity[i] = np.mean(tc_grid1 == tc_grid2)

    # If no neighboring parts are similar, return the initial set of parts
    # This fast-path avoids the merge loop when all parts are already distinct
    if np.all(part_similarity < cutoff):
        return parts_init, np.ones(nparts) * 8

    # Merge parts in order of appearance. Parts can only be merged once.
    # A greedy left-to-right scan: when a similar pair is found, both are
    # consumed (i advances by 2) so neither can be re-merged with its next neighbour
    i = 0
    parts_final = []
    part_nbars = []
    while i < (nparts):
        # For the last part, we check if it was merged, and if not we add it
        # When i points to the final chunk, look back at the previous similarity
        # score to determine whether it was already consumed by a merge
        if i == nparts - 1:
            if part_similarity[i-1] < cutoff:
                # Previous pair was not merged, so this last chunk is still unpaired
                parts_final.append(parts_init[i])
                part_nbars.append(8)
                i += 1
            else:
                # Previous pair was merged, so this chunk was already consumed; skip it
                i += 1
        else:
            if part_similarity[i] >= cutoff:
                # Adjacent parts are similar enough to be considered the same structural
                # part played twice; concatenate them into one 16-bar entry
                parts_final.append((np.concatenate([parts_init[i][0], parts_init[i+1][0]]),
                                    np.concatenate([parts_init[i][1], parts_init[i+1][1]])))
                part_nbars.append(16)
                i += 2
            else:
                parts_final.append(parts_init[i])
                part_nbars.append(8)
                i += 1

    return parts_final, np.array(part_nbars)


### Given a tune_id, load all settings and split them into parts
def get_all_parts_tune(df, tunes, tune_id, min_bars=8):
    """
    Screen and segment all settings of a single tune into structural parts.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with at least the columns ``tune_id`` and
        ``setting_id``.
    tunes : dict
        Mapping from ``setting_id`` to a setting dict (with keys ``'bars'``,
        ``'dur'``, ``'tmidi'``).
    tune_id : int
        The tune family identifier whose settings should be processed.
    min_bars : int, optional
        Part length in bars passed through to ``screen_bars`` and
        ``separate_tune_into_parts``.  Default is ``8``.

    Returns
    -------
    all_parts : list of tuple
        Each element is the return value of ``separate_tune_into_parts`` for
        one accepted setting: ``(parts_final, part_nbars)``.
    to_keep : list of int
        Zero-based indices (within the ordered list of settings for
        ``tune_id``) of the settings that passed screening and are represented
        in ``all_parts``.

    Notes
    -----
    Settings that fail ``screen_bars`` — because they contain bars with
    incorrect durations or a bar count that is not a multiple of ``min_bars``
    after pickup removal — are silently excluded.  ``to_keep`` enables the
    caller to map entries in ``all_parts`` back to the original setting list.
    """
    all_parts = []
    to_keep = []
    for k, setting_id in enumerate(df.loc[df.tune_id==tune_id, 'setting_id']):
        # Remove bars that do not add up to the expected duration for the given meter.
        # This does two things:
        #   Removes pickups
        #   Removes any bars with incorrect durations (rare, but it happens)
        # If any bars with incorrect durations are removed, then the total number of bars
        # will not be a multiple of 8, and it will be discarded
        use_okay, bar_dur, bar_tmidi = screen_bars(tunes[setting_id], min_bars=min_bars)

        if use_okay:
            all_parts.append(separate_tune_into_parts(bar_tmidi, bar_dur, min_bars=min_bars))
            to_keep.append(k)
    return all_parts, to_keep


### Load all parts from TheSession tunes
### (code takes about 2 hours to run)
def get_all_parts_thesession(df, tunes, redo=False):
    """
    Build (or load from cache) the complete parts inventory for all tunes in
    TheSession dataset.

    Results are cached as two pickle files under ``PATH_CACHE``; subsequent
    calls with ``redo=False`` simply deserialise and return the cache, making
    the typical call near-instant.  A full rebuild takes approximately 2 hours.

    The function iterates over every tune family in ``df``, screens and
    segments each setting via ``get_all_parts_tune``, and assigns each
    resulting part a globally unique ``part_id`` of the form
    ``"{tune_id}_{setting_id}_{part_index}"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with at least the columns ``tune_id`` and
        ``setting_id``.
    tunes : dict
        Mapping from ``setting_id`` to a setting dict (with keys ``'bars'``,
        ``'dur'``, ``'tmidi'``).
    redo : bool, optional
        If ``True``, discard any cached results and recompute from scratch.
        Default is ``False``.

    Returns
    -------
    df_parts : pandas.DataFrame
        One row per extracted part with columns:

        * ``part_id``   — unique string key ``"{tune_id}_{setting_id}_{part_no}"``
        * ``tune_id``   — integer tune family identifier
        * ``setting_id``— integer setting identifier
        * ``part_no``   — zero-based index of the part within its setting
        * ``num_parts`` — total number of *settings* (not parts) retained for
          this tune family after screening
    parts_out : dict
        Maps each ``part_id`` to a tuple
        ``((dur_array, tmidi_array), nbars)`` where ``dur_array`` and
        ``tmidi_array`` are the concatenated note-level arrays for the part
        and ``nbars`` is the bar count (8 or 16).

    Notes
    -----
    ``num_parts`` in ``df_parts`` records the count of accepted *settings* for
    the parent tune, not the number of parts within a single setting.  This
    reflects how many settings survived ``screen_bars`` and is useful for
    filtering tune families with sparse representation.

    The cache files are:

    * ``PATH_CACHE / "all_parts_thesession_df.pkl"`` — the DataFrame
    * ``PATH_CACHE / "all_parts_thesession.pkl"``    — the ``parts_out`` dict
    """
    path_df = PATH_CACHE.joinpath("all_parts_thesession_df.pkl")
    path = PATH_CACHE.joinpath("all_parts_thesession.pkl")
    # Return cached results immediately if both files are present and redo is not requested
    if all([path_df.exists(), path.exists()]) and not redo:
        return pd.read_pickle(path_df), pickle.load(open(path, 'rb')),
    else:
        cols = ["part_id", "tune_id", "setting_id", "part_no", 'num_parts']
        rows = []
        parts_out = {}
        ts = time()
        for tune_id in tqdm(df.tune_id.unique()):
#       for tune_id in df.tune_id.unique():
            # Retrieve the ordered array of setting IDs for this tune family so
            # that to_keep indices can be mapped back to the correct setting_id
            settings = df.loc[df.tune_id==tune_id, 'setting_id'].values
            all_parts, to_keep = get_all_parts_tune(df, tunes, tune_id)
            # j is the index into `settings`; parts is (parts_final, part_nbars)
            for j, parts in zip(to_keep, all_parts):
                # k is the zero-based part index within this setting
                for k, (part, nbars) in enumerate(zip(*parts)):
                    part_id = f"{tune_id}_{settings[j]}_{k}"
                    # Data structure of "p": (parts (dur, tmidi), nbars_per_part)
                    parts_out[part_id] = (part, nbars)
                    # num_parts records how many settings survived screening for this tune
                    rows.append([part_id, tune_id, settings[j], k, len(to_keep)])
        print("Time taken: ", (time() - ts) / 60)
        pickle.dump(parts_out, open(path, 'wb'))
        df_parts = pd.DataFrame(data=rows, columns=cols)
        return df_parts, parts_out


#ef update_parts_df(df_parts):



#######################################################
### Evaluate part separation algorithm


### Create a template json file for annotating ground truth
def save_gt_part_template(df, tune_id):
    """
    Write a blank JSON annotation template for manual ground-truth labelling
    of part boundaries.

    The file is written to
    ``PATH_DATA / "TheSession-data/part_annotations/tune_{tune_id}.json"``
    and is skipped if that file already exists to protect completed
    annotations from accidental overwriting.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with at least the columns ``tune_id`` and
        ``setting_id``.
    tune_id : int
        Tune family for which the template should be created.

    Returns
    -------
    None
        Prints a warning if the file already exists; otherwise writes the
        template silently.

    Notes
    -----
    Each entry in the JSON list has the form
    ``{"setting_id": <int>, "part_start": {}}``.
    The ``part_start`` dict is intentionally empty; the annotator should
    populate it with ``{part_label: bar_index, ...}`` entries.

    The JSON is pretty-formatted so that each ``}`` delimiter is followed by
    a newline, making the file easier to edit by hand.
    """
    path = PATH_DATA.joinpath("TheSession-data/part_annotations", f'tune_{tune_id}.json')
    if path.exists():
        print("File already exists!")
    else:
        setting_id_list = df.loc[df.tune_id==tune_id, 'setting_id'].values
        out = [{"setting_id":int(ID), "part_start":{}} for ID in setting_id_list]
        with open(path, 'w') as o:
            # Insert a newline after each closing brace to improve readability
            o.write('},\n'.join(json.dumps(out).split('},')))


### Load annotated ground truth for number of parts in settings
def load_gt_parts(tune_id):
    """
    Load a previously saved ground-truth annotation file for a tune family.

    Parameters
    ----------
    tune_id : int
        Tune family identifier; the file is expected at
        ``PATH_DATA / "TheSession-data/part_annotations/tune_{tune_id}.json"``.

    Returns
    -------
    gt : list of dict
        Parsed JSON content: a list where each element is a dict with keys
        ``'setting_id'`` (int) and ``'part_start'`` (dict mapping part labels
        to bar indices).
    """
    path = PATH_DATA.joinpath("TheSession-data/part_annotations", f'tune_{tune_id}.json')
    gt = json.load(open(path, 'r'))
    return gt


### Evaluate part separation algorithm for a tune family
def evaluate_nparts(df, tunes, tune_id, cutoff=0.8, min_bars=8):
    """
    Compute the part-count accuracy of the separation algorithm for one tune
    family against manually annotated ground truth.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with at least the columns ``tune_id`` and
        ``setting_id``.
    tunes : dict
        Mapping from ``setting_id`` to a setting dict.
    tune_id : int
        Tune family to evaluate.
    cutoff : float, optional
        Chroma-similarity merge threshold passed to
        ``separate_tune_into_parts``.  Default is ``0.8``.
    min_bars : int, optional
        Part length in bars.  Default is ``8``.

    Returns
    -------
    acc : float
        Fraction of settings (among those that passed screening) for which
        the algorithm predicted the correct number of structural parts.

    Notes
    -----
    Settings that fail ``screen_bars`` are assigned a predicted part count of
    ``0`` and are excluded from the accuracy calculation via the ``idx``
    mask.  This ensures that transcription errors do not penalise the
    algorithm's score.

    Only settings that appear in both the metadata table and the ground-truth
    annotation file are evaluated; settings present in one source but not the
    other are silently skipped.
    """
    setting_list = df.loc[df.tune_id==tune_id, 'setting_id'].values

    # Load ground-truth
    gt = load_gt_parts(tune_id)
    # Filter ground-truth to only the settings that also appear in the metadata
    gt_settings = np.array([x['setting_id'] for x in gt if x['setting_id'] in setting_list])
    gt_nparts = np.array([len(x['part_start']) for x in gt if x['setting_id'] in setting_list])

    # Calculate the number of parts per tune
    nparts = []
    for s in setting_list:
        if s not in gt_settings:
            continue
        use_okay, bar_dur, bar_tmidi = screen_bars(tunes[s], min_bars=min_bars)
        if use_okay:
            nparts.append(len(separate_tune_into_parts(bar_tmidi, bar_dur, cutoff)[0]))
        else:
            # Assign 0 as a sentinel for settings that could not be processed;
            # these are excluded from the accuracy calculation below
            nparts.append(0)
    nparts = np.array(nparts)

    # Some tunes have errors. Don't include these in the accuracy calculation
    # idx masks out the sentinel-zero entries so they do not lower the score
    idx = nparts != 0
    acc = np.mean(nparts[idx] == gt_nparts[idx])
    return acc


### Evaluate part separation algorithm across all tune families
def evaluate_nparts_params(df, tunes):
    """
    Grid-search the ``cutoff`` parameter of ``separate_tune_into_parts``
    across a fixed set of tune families.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with at least the columns ``tune_id`` and
        ``setting_id``.
    tunes : dict
        Mapping from ``setting_id`` to a setting dict.

    Returns
    -------
    acc : numpy.ndarray
        2-D array of shape ``(len(tune_list), len(cutoff_list))`` where
        ``acc[i, j]`` is the part-count accuracy for tune family
        ``tune_list[i]`` at similarity threshold ``cutoff_list[j]``.

    Notes
    -----
    The evaluated tune families are ``[2, 21, 27, 34, 62, 74]`` and the
    cutoff values sweep from ``0.50`` to ``0.95`` in steps of ``0.05``.
    These represent a representative cross-section of Irish tune types
    (reels, jigs, hornpipes) used to calibrate the merge threshold.
    """
    tune_list = [2, 21, 27, 34, 62, 74]
    cutoff_list = np.arange(0.5, 1, 0.05)
    acc = []
    for t in tune_list:
        for cutoff in cutoff_list:
            acc.append(evaluate_nparts(df, tunes, t, cutoff=cutoff))
    return np.array(acc).reshape(len(tune_list), len(cutoff_list))


