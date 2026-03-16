import json
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesession.config import *
from thesession.io import tune_parser as TP
from thesession import utils



######################################################################
### thesession data


def load_thesession_tunes(redo=False):
    """
    Load the filtered TheSession tune-level dataset used for ROC curve
    analysis and parameter optimisation.

    Builds on the full pyabc-parsed dataset (``thesession_full.pkl``) and
    applies a post-hoc filter that removes tunes with grace notes,
    polyphonic pitch, or multiple voices.  The result is cached as
    ``thesession_tunes.pkl``.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, ignore the cache and reprocess from scratch.
        Default is ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Filtered tune-level metadata table.
    """
    path = PATH_CACHE.joinpath("thesession_tunes.pkl")
    if path.exists() and not redo:
        return pd.read_pickle(path)
    df, json_data = load_thesession_data_raw()
    df = process_thesession_tunes_pyabc(df, json_data, full=True)
    df = df.loc[~(df.has_grace | df.has_poly | df.has_voices)]
    df.to_pickle(path)
    return df


### Main function to load thesession data
### Runs preliminary pipeline if there is no cached data,
### or if redo=True
def load_thesession_data(redo=False):
    """
    Load the fully cleaned and processed TheSession dataset.

    Returns cached results if available; otherwise runs the full
    processing pipeline via ``process_thesession_tunes``.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, ignore existing cache files and reprocess from
        scratch.  Default is ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Setting-level metadata table (one row per setting).
    data : dict
        Mapping from ``setting_id`` to the music21-derived feature
        dict (keys: ``'bar_onset'``, ``'onsets'``, ``'dur'``, ``'midi'``,
        ``'rests'``, ``'bars'``, ``'tmidi'``, ``'tchroma'``).
    """
    path_df = PATH_CACHE.joinpath("thesession_cleaned_processed.pkl")
    path_data = PATH_CACHE.joinpath("thesession_music21.pkl")
    if (path_df.exists() and path_data.exists()) and not redo:
        df = pd.read_pickle(path_df)
        data = pickle.load(open(path_data, 'rb'))
        return df, data
    else:
        return process_thesession_tunes(redo)


### Load the raw data from thesession downloaded from github
def load_thesession_data_raw():
    """
    Load the raw TheSession CSV metadata and JSON tune data from disk.

    The files are read from the paths defined in ``config.py``:
    ``PATH_DATA / "TheSession-data/csv/tunes.csv"`` and
    ``PATH_DATA / "TheSession-data/json/tunes.json"``.

    Returns
    -------
    df : pandas.DataFrame
        Raw metadata table as downloaded from the TheSession GitHub
        repository.
    json_data : list of dict
        List of tune dicts in TheSession JSON format; indices align with
        the rows of ``df``.
    """
    df = pd.read_csv(PATH_DATA.joinpath("TheSession-data", "csv", "tunes.csv"))
    json_data = json.load(open(PATH_DATA.joinpath("TheSession-data", "json", "tunes.json"), 'r'))
    return df, json_data


### Process all of thesession data, to produce a final version
def process_thesession_tunes(redo=False):
    """
    Run the full TheSession processing pipeline and cache the results.

    The pipeline performs the following steps:

    1. Load raw CSV/JSON data.
    2. Parse ABC notation with pyabc (``process_thesession_tunes_pyabc``),
       computing pitch arrays and filtering out problematic tunes.
    3. Parse ABC notation with music21 (``process_thesession_tunes_music21``),
       extracting bar boundaries, onsets, durations, and MIDI pitches.
    4. Update the music21 dict with key-transposed pitch arrays and remove
       any settings that failed to parse (``update_music21_data``).
    5. Drop settings absent from the music21 dict.
    6. Infer the modal character of each setting from its pitch-class
       histogram.
    7. Cache the final DataFrame and tune dict to ``PATH_CACHE``.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, force recomputation of each intermediate step.
        Default is ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned and annotated setting metadata.
    tunes : dict
        Mapping from ``setting_id`` to the fully annotated feature dict.
    """
    # Load raw data
    df, json_data = load_thesession_data_raw()

    # Process with pyabc
    df = process_thesession_tunes_pyabc(df, json_data, redo=False, full=False)

    # Process with music21
    tunes = process_thesession_tunes_music21(df, json_data, redo=False)

    # Update the music21 data
    tunes = update_music21_data(df, tunes)

    # Remove all tunes that music21 did not parse
    df = df.loc[df.setting_id.apply(lambda x: x in tunes)]
    print(f"Tunes processed by music21. {len(df)} tunes left after cleaning...")

    # Statistically infer the most likely mode, given the tonal hierachy
    df['inferred_mode'] = df['setting_id'].apply(lambda x: utils.check_mode(tunes[x]['tchroma']))

    path_df = PATH_CACHE.joinpath("thesession_cleaned_processed.pkl")
    path_data = PATH_CACHE.joinpath("thesession_music21.pkl")
    df.to_pickle(path_df)
    pickle.dump(tunes, open(path_data, 'wb'))

    return df, tunes


### Primary processing of thesession data, using pyabc and custom code
### This stage is required for filtering out problematic tunes
def process_thesession_tunes_pyabc(df, json_data, redo=False, full=False):
    """
    Parse and filter TheSession tunes using pyabc, adding pitch and
    structural annotation columns to ``df``.

    The function reads each tune's ABC string via ``TP.parse_thesession_tune``
    (pyabc backend), appends columns for pitch, key, ties, grace notes,
    polyphony, multiple voices, and repeat consistency, and then optionally
    discards tunes that cannot be safely handled by music21.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw metadata table from ``load_thesession_data_raw``.
    json_data : list of dict
        Parallel list of tune dicts from ``load_thesession_data_raw``.
    redo : bool, optional
        If ``True``, ignore any cached result and reparse.  Default is
        ``False``.
    full : bool, optional
        If ``True``, keep all successfully parsed tunes (including those
        with grace notes, polyphony, or repeat issues).  If ``False``
        (default), apply strict filtering so that only clean tunes
        compatible with music21 are retained.

    Returns
    -------
    df : pandas.DataFrame
        Annotated and (if ``full=False``) filtered DataFrame, also
        persisted to ``PATH_CACHE``.

    Notes
    -----
    Added columns include: ``notestr``, ``abspitch``, ``key``,
    ``has_keys``, ``tchroma``, ``tchroma_octave``, ``mel_len``,
    ``key_change``, ``has_key_change``, ``has_grace``, ``has_poly``,
    ``has_voices``, ``repeats_consistent``.

    Tunes that raise an exception during parsing are silently dropped;
    their indices are collected in ``idx_ignored``.
    """
    if full:
        path = PATH_CACHE.joinpath("thesession_full.pkl")
    else:
        path = PATH_CACHE.joinpath("thesession_clean.pkl")

    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        print(f"Starting to parse TheSession. {len(df)} tunes to start with...")
        parsed_data = TP.parse_thesession_tune(json_data[0])
        for k in parsed_data.keys():
            df[k] = pd.Series(dtype=object)

        idx_keep = []
        idx_ignored = []
        for i, data in tqdm(zip(df.index, json_data)):
            try:
                parsed_data = TP.parse_thesession_tune(data)
                for k, v in parsed_data.items():
                    df.at[i, k] = v
                idx_keep.append(i)
            except Exception as e:
                idx_ignored.append(i)

        # Remove indices of tunes that weren't processed
        df = df.loc[idx_keep]
        print(f"Tunes processed. {len(df)} tunes were successfully processed...")

        # Transpose + convert to chroma
        # tchroma is the key-transposed pitch class; tchroma_octave retains octave
        df['tchroma'] = (df['abspitch'] - df['key']) % 12
        df['tchroma_octave'] = (df['abspitch'] - df['key'])

        # Update with melody length
        df['mel_len'] = df['tchroma'].apply(len)

        # Check for key changes
        # Inline key changes like [K:Gmaj] indicate modulation mid-tune
        pattern_key = r'\[K:[^\]]+\]'
        df['key_change'] = df['abc'].apply(lambda x: re.findall(pattern_key, x))
        df['has_key_change'] = df['key_change'].apply(len) > 0

        # Check for grace notes
        # Grace notes in ABC notation are enclosed in curly braces: {A}
        pattern_grace = r'\{[^}]+\}'
        df['has_grace'] = df['abc'].apply(lambda x: len(re.findall(pattern_grace, x)) > 0)

        # Check for polyphonic notes (within a single voice)
        # Chord-like groups in ABC are enclosed in square brackets: [EG]
        pattern_poly = r'\[[^\]:]*\]'
        df['has_poly'] = df['abc'].apply(lambda x: len(re.findall(pattern_poly, x)) > 0)

        # Check for multiple voices
        # Voice declarations appear as V:1, V:2, etc.
        pattern_voice = r'V:\d'
        df['has_voices'] = df['abc'].apply(lambda x: len(re.findall(pattern_voice, x)) > 0)

        # Check for consistency of repeat lines
        df['repeats_consistent'] = df['abc'].apply(TP.check_repeat_consistency)

        # Remove any tunes that music21 won't be able to parse properly
        if not full:
            df = df.loc[~(df.has_grace | df.has_poly | df.has_voices) & (df.repeats_consistent)]
            print(f"Tunes processed. {len(df)} tunes left after cleaning...")

        df.to_pickle(path)

        return df


### Primary extraction of melodic information, using music21
### Produces a dictionary, with setting_id as keys
def process_thesession_tunes_music21(df, json_data, redo=False):
    """
    Extract bar-level melodic features from ABC notation using music21.

    Iterates over all settings in ``df``, converts each to a music21
    score via ``TP.parse_thesession_tune(alg='music21')``, and collects
    bar boundaries, onsets, durations, and MIDI pitches.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered metadata table (output of
        ``process_thesession_tunes_pyabc``); provides ``index`` and
        ``setting_id`` for iteration.
    json_data : list of dict
        Parallel list of raw tune dicts; indexed by ``df.index``.
    redo : bool, optional
        If ``True``, ignore any cached result and re-extract.  Default is
        ``False``.

    Returns
    -------
    out : dict
        Mapping from ``setting_id`` to a dict with keys ``'bar_onset'``,
        ``'onsets'``, ``'dur'``, ``'midi'``, ``'rests'``, and ``'bars'``.
        Tunes that raise exceptions during music21 parsing are silently
        omitted.
    """
    path = PATH_CACHE.joinpath("thesession_music21.pkl")
    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        out = {}
        for i, setting in tqdm(zip(df.index, df.setting_id)):
            data = json_data[i]
            try:
                out[setting] = TP.parse_thesession_tune(data, alg='music21')
            except Exception as e:
                print(e)
        pickle.dump(out, open(path, 'wb'))
        return out


### Update the dictionaries with extra information.
### This requires information about the musical key
### that was obtained from the ABC header.
### It also performs a consistency check to make sure
### tunes were parsed correctly
def update_music21_data(df, tunes):
    """
    Enrich the music21 feature dicts with key-transposed pitches and
    remove settings with data-integrity errors.

    For each setting present in both ``df`` and ``tunes``, this function:

    * Converts ``'bar_onset'``, ``'onsets'``, ``'dur'``, and ``'midi'``
      arrays to ``float``.
    * Checks that ``onsets`` and ``midi`` have the same length (a sign
      of a parsing failure); removes the setting if they differ.
    * Adds ``'tmidi'`` (key-transposed MIDI pitch) and ``'tchroma'``
      (pitch class) derived from the key stored in ``df``.

    Settings whose ``setting_id`` is absent from ``df`` are also removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered and annotated metadata table with at minimum the
        columns ``setting_id`` and ``key``.
    tunes : dict
        Mapping from ``setting_id`` to the music21 feature dict as
        returned by ``process_thesession_tunes_music21``.

    Returns
    -------
    tunes : dict
        The same dict with invalid entries removed and new keys
        ``'tmidi'`` and ``'tchroma'`` added to each remaining entry.
    """
    setting_idx = {s:i for i, s in zip(df.index, df.setting_id)}
    keys2del = []
    for k, v in tunes.items():
        if k in setting_idx:
            for k2 in ['bar_onset', 'onsets', 'dur', 'midi']:
                v[k2] = np.array(v[k2], float)
            if v['onsets'].size != v['midi'].size:
                print('Size mismatch! ', k)
                keys2del.append(k)
            # Transpose absolute MIDI pitch by the tonic key to get relative (transposed) pitch
            v['tmidi'] = v['midi'] - df.loc[setting_idx[k], 'key']
            v['tchroma'] = v['tmidi'] % 12
            tunes[k] = v
        else:
            keys2del.append(k)
    for k in keys2del:
        del tunes[k]
    return tunes


###############################################################
### Meertens data


### Meertens data is distributed across many files.
### This function finds all the file paths.
def load_meertens_paths():
    """
    Discover all Kern files in the Meertens MTC-FS-INST-2.0 corpus.

    Returns
    -------
    numpy.ndarray, shape (n_files, 3)
        Each row contains ``[path, song_id, variant]`` where ``path`` is
        a ``pathlib.Path`` object, ``song_id`` is the base identifier, and
        ``variant`` is the variant suffix extracted from the file stem
        (format: ``"{song_id}_{variant}"``).
    """
    path_list = sorted(PATH_DATA.joinpath("Meertens", "MTC-FS-INST-2.0/krn").glob("*"))
    # File stems have the format "{song_id}_{variant}"; split on the last underscore
    song_id, variant = np.array([p.stem.split('_') for p in path_list]).T
    return np.array([path_list, song_id, variant]).T


### Load tune type (vocal / instrumental) annotations
def load_meertens_metadata(df):
    """
    Attach ``'type'`` (vocal / instrumental) annotations to the Meertens
    DataFrame from the MTC-FS-INST-2.0 metadata CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Meertens DataFrame with at least a ``'ref'`` column matching the
        ``'filename'`` field in the metadata CSV.

    Returns
    -------
    df : pandas.DataFrame
        The same DataFrame with a new ``'type'`` column.
    """
    cols = pd.read_csv(PATH_DATA.joinpath('Meertens/MTC-FS-INST-2.0/metadata/MTC-FS-INST-2.0-fieldnames.csv')).columns
    dfm = pd.read_csv(PATH_DATA.joinpath('Meertens/MTC-FS-INST-2.0/metadata/MTC-FS-INST-2.0.csv'), names=cols)
    key = {f:t for f, t in zip(dfm['filename'], dfm['type'])}
    df['type'] = df['ref'].map(key)
    return df



### Load meertens data:
### Creates a dataframe for metadata and summary stats,
### and a dictionary for the tune sequences
def load_meertens_data(redo=False):
    """
    Load (or build) the Meertens MTC-FS-INST-2.0 dataset.

    On first run (or when ``redo=True``), discovers all Kern files,
    parses each with music21 via ``TP.parse_kern_music21``, and attaches
    key and pitch-class columns.  Results are cached to ``PATH_CACHE``
    for subsequent calls.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, ignore any cached files and rebuild.  Default is
        ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Metadata table with columns ``path``, ``song_id``, ``variant``,
        ``ref``, ``type``, ``key``, ``tchroma_octave``, and ``tchroma``.
    data : dict
        Mapping from row index to the music21 feature dict for each tune
        (keys: ``'bar_onset'``, ``'onsets'``, ``'dur'``, ``'midi'``,
        ``'rests'``, ``'bars'``, ``'key'``, ``'accidentals'``).
    """
    path_df = PATH_CACHE.joinpath("meertens_summary.pkl")
    path_data = PATH_CACHE.joinpath("meertens_tunes.pkl")
    if path_df.exists() and not redo:
        df = pd.read_pickle(path_df)
        data = pickle.load(open(path_data, 'rb'))
    else:
        df = pd.DataFrame(data=load_meertens_paths(), columns=['path', 'song_id', 'variant'])
        df['ref'] = [f"{a}_{b}" for a, b in zip(df.song_id, df.variant)]
        df = load_meertens_metadata(df)

        data = {}
        for i, p in tqdm(zip(df.index, df.path)):
            data[i] = TP.parse_kern_music21(p)

        df = update_meertens_df(df, data)

        df.to_pickle(path_df)
        pickle.dump(data, open(path_data, 'wb'))
    return df, data


### Updates the meertens dataframe
def update_meertens_df(df, data):
    """
    Add key, transposed-pitch, and pitch-class columns to the Meertens
    DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Meertens metadata DataFrame (from ``load_meertens_paths`` /
        ``load_meertens_metadata``).
    data : dict
        Mapping from row index to the music21 feature dict for each tune.

    Returns
    -------
    df : pandas.DataFrame
        The same DataFrame with new columns ``'ref'``, ``'key'``,
        ``'tchroma_octave'``, and ``'tchroma'``.
    """
    df['ref'] = df.path.apply(lambda x: x.stem)
    # get_key_idx maps the key string (e.g. 'G') to an integer semitone offset from C
    df['key'] = [TP.get_key_idx(data[i]['key']) for i in df.index]
    df['tchroma_octave'] = [(data[i]['midi'] - k) for i, k in zip(df.index, df.key)]
    df['tchroma'] = [(data[i]['midi'] - k) % 12 for i, k in zip(df.index, df.key)]
    return df



######################################################################
### Bronson


### Load Bronson data
def load_bronson_data(redo=False):
    """
    Load the Bronson (British/American folk song) dataset from a
    pre-processed pickle file.

    The function reads the merged summary pickle, computes
    key-transposed MIDI pitch (``tmidi``), and adds ``tchroma_octave``
    and ``tchroma`` columns.  Rows where ``tmidi`` is null (failed
    transposition) are dropped.

    Parameters
    ----------
    redo : bool, optional
        Not used; included for API consistency with other loaders.
        Default is ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned Bronson DataFrame with columns ``tmidi``,
        ``tchroma_octave``, and ``tchroma`` added.
    """
    path = PATH_DATA.joinpath('Bronson/merged_summary.pkl')
    df = pd.read_pickle(path)
    # Transpose absolute MIDI pitch by the stored key value
    df['tmidi'] = df['midi'] - df['key']
    df = df.loc[df.tmidi.notnull()]
    df['tchroma_octave'] = [a.astype(int) for a in df['tmidi']]
    df['tchroma'] = [a.astype(int) for a in df['tmidi'] % 12]
    return df
