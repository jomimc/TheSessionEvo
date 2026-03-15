from collections import Counter, defaultdict
from pathlib import Path
import pickle

import music21
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.analysis import optimization as OP
from thesession.alignment import pairwise as seq_align
from thesession.analysis import substitution as SM
from thesession.io import tune_parser as TP
from thesession import utils


### Load the set of manually-aligned sequence pairs from Savage et al.
def load_savage_df(full=False, redo=False, keep_japanese=False):
    """
    Load the Savage et al. manually-aligned melody dataset as a DataFrame.

    Parameters
    ----------
    full : bool, optional
        If True, load the full-song dataset (``MelodicEvoSeqFullSongs.xlsx``);
        otherwise load the partial dataset (``MelodicEvoSeq.xlsx``).
        Default False.
    redo : bool, optional
        If True, ignore any cached pickle and rebuild from the Excel source.
        Default False.
    keep_japanese : bool, optional
        When loading the partial dataset, retain Japanese-language entries if
        True; otherwise restrict to English entries only.  Has no effect when
        ``full=True``.  Default False.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``PairNo``, ``name``, ``Language``, ``chapter``,
        ``song_ref``, ``PID``, ``seq_aligned``, ``seq_unaligned``,
        ``ref`` (chapter_songref composite), ``tchroma`` (integer pitch-class
        array derived from the unaligned sequence).  Rows with unparseable
        note codings are silently dropped.

    Notes
    -----
    Results are cached as a pickle in ``PATH_CACHE`` so subsequent calls are
    fast.  Pass ``redo=True`` to force a fresh parse from the Excel file.
    The ``tchroma`` column uses the ``note_map`` lookup from ``config`` to
    convert Savage's character-based note coding to integer pitch-class values.
    """
    if full:
        path = PATH_CACHE.joinpath("savage_full.pkl")
    else:
        path = PATH_CACHE.joinpath("savage_part.pkl")

    if path.exists() and not redo:
        return pd.read_pickle(path)

    if full:
        df = pd.read_excel(PATH_DATA.joinpath('Bronson/MelodicEvoSeqFullSongs.xlsx'), sheet_name='MelodicEvoSeq')
    else:
        df = pd.read_excel(PATH_DATA.joinpath('Bronson/MelodicEvoSeq.xlsx'), sheet_name='MelodicEvoSeq')
        if not keep_japanese:
            df = df.loc[df.Language=='English']

    # Rename columns and drop unwanted columns
    old_col = ['PairNo', 'Song title', 'Language', 'Child Ballad no./NHK Volume no.',
               'Variant no.', 'PID', 'Full note sequence (aligned)', 'Full note sequence (unaligned)']
    new_col = ['PairNo', 'name', 'Language', 'chapter', 'song_ref', 'PID', 'seq_aligned', 'seq_unaligned']
    df = df.loc[:, old_col]
    df = df.rename(columns={c1:c2 for c1, c2 in zip(old_col, new_col)})

    # Remove Addenda names ('App') from chapters
    # And converts whitespace to underscore (necessary for mmseqs)
    if full:
        df['chapter'] = df['chapter'].apply(lambda x: x.replace('App', '').replace(" ", "_") if isinstance(x, str) else str(x))

    # Create a unique reference field
    df['ref'] = [f"{str(c)}_{str(s)}" for c, s in zip(df['chapter'], df['song_ref'])]

    # Convert Pat's codings to integer values
    idx2drop = [] # Some of the codings in Savage et al are wrong, so keep a list of these
    tchroma = []
    for i, s in zip(df.index, df['seq_unaligned']):
        try:
            tchroma.append(np.array([note_map[n] for n in s.strip()]))
        except:
            # Row has an unrecognised note character — skip it
            idx2drop.append(i)
    df = df.drop(index=idx2drop)
    df['tchroma'] = tchroma

    # Save to pickle
    df.to_pickle(path)

    return df


def load_checked_subset():
    """
    Load the manually-verified subset of Savage–Fitch matched song pairs.

    Returns
    -------
    df : pd.DataFrame
        The matched-songs CSV enriched with two ``tchroma`` columns:
        ``tchroma_savage`` (pitch-class sequence from the Savage dataset) and
        ``tchroma_fitch`` (pitch-class sequence parsed from the Bronson/Fitch
        dataset, transposed to a common tonic).  Also includes boolean columns
        ``len_same`` and ``exact_same`` indicating sequence-level agreement.

    Notes
    -----
    Rows where ``IndexFitch`` is null are dropped before merging.  The Fitch
    sequences are transposed by subtracting the stored key value (mod 12) so
    that both sequences share the same pitch-class zero reference.
    """
    dfb = load_tunes.load_bronson_data()
    dfs = load_savage_df()

    df = pd.read_csv('../savage_matched_songs.csv')
    df = df.loc[df.IndexFitch.notnull()]
    df['IndexFitch'] = df['IndexFitch'].astype(int)

    df['tchroma_savage'] = dfs.loc[df.Index, 'tchroma'].values

    key = dfb.loc[df.IndexFitch, 'key']
    midi = []
    for i in df.IndexFitch:
        data = dfb.loc[i].to_dict()
        data['key'] = chromatic_notes[int(data['key'])]
        midi.append(TP.parse_thesession_tune(data, 'music21', expandRepeats=False)['midi'])
    # Transpose each Fitch melody to tonic-relative pitch classes (mod 12)
    df['tchroma_fitch'] = [((np.array(m) - k) % 12).astype(int) for m, k in zip(midi, key)]

    df['len_same'] = [len(x) == len(y) for x, y in zip(df['tchroma_savage'], df['tchroma_fitch'])]
    df['exact_same'] = [len(x) == len(y) and np.all(x == y) for x, y in zip(df['tchroma_savage'], df['tchroma_fitch'])]

    return df


### Load the full set of sequence pairs from Savage et al.
def load_df_full():
    """
    Load the raw full-song Excel sheet from Savage et al. without any processing.

    Returns
    -------
    df : pd.DataFrame
        The ``MelodicEvoSeq`` sheet from ``MelodicEvoSeqFullSongs.xlsx`` as-is.
    """
    return pd.read_excel(PATH_DATA.joinpath('Bronson/MelodicEvoSeqFullSongs.xlsx'), sheet_name='MelodicEvoSeq')


### Given a sequence, and an aligned version (i.e. the sequence plus gaps),
### replace the characters in the aligned version with the characters from
### the given sequence
def convert_seq(tchroma, aligned):
    """
    Re-encode an aligned sequence string using integer pitch-class characters.

    Parameters
    ----------
    tchroma : array-like of int
        Unaligned pitch-class sequence (gaps excluded).
    aligned : str
        Aligned version of the sequence using Savage's character coding, with
        ``'-'`` characters marking insertion/deletion positions.

    Returns
    -------
    seq : str
        The aligned string with Savage characters replaced by the single-char
        tokens produced by ``utils.tchroma2seq``, preserving gap positions.

    Notes
    -----
    Gaps are located in ``aligned``, then re-inserted at the same positions
    into the re-encoded unaligned sequence so the alignment structure is
    preserved.
    """
    gaps = np.where(np.array(list(aligned)) == '-')[0]
    seq = ''.join(utils.tchroma2seq(tchroma))
    # Re-insert gap characters at their original positions
    for i in gaps:
        seq = seq[:i] + '-' + seq[i:]
    return seq


### For each sequence pair in Savage et al, run local alignment algorithm,
### and check if the manually-aligned alignments are one of the top-scoring
### alignments
def compare_all_alignments(df, **kwargs):
    """
    Check whether Savage's manual alignments are recovered by the pairwise aligner.

    For every pair in ``df``, runs ``get_pairwise_nhits`` and tests whether the
    manually-curated alignment appears among the optimal solutions returned.

    Parameters
    ----------
    df : pd.DataFrame
        Savage dataset as returned by ``load_savage_df``, containing at least
        ``PairNo``, ``tchroma``, and ``seq_aligned`` columns.
    **kwargs
        Passed directly to ``seq_align.get_pairwise_nhits`` (e.g. scoring
        parameters).

    Returns
    -------
    is_found : np.ndarray of bool
        One entry per unique pair number; True if the manual alignment was
        found among the top alignments.
    num_align : np.ndarray of int
        Number of optimal alignments returned by the aligner for each pair.
    """
    pair_list = np.array(sorted(df['PairNo'].unique()))
    is_found = []
    num_align = []
    for pair in pair_list:
        idx = df.loc[df['PairNo'] == pair].index
        # Re-encode both sequences with our character set before aligning
        al1, al2 = [convert_seq(*df.loc[i, ['tchroma', 'seq_aligned']]) for i in idx]
        s1, s2 = [utils.tchroma2seq(df.loc[i, 'tchroma']) for i in idx]
        alignments = seq_align.get_pairwise_nhits(s1, s2, **kwargs)
#       print(f"Pair {pair}. Checking {len(alignments)} alignments...")
        is_found.append(is_alignment_found(al1, al2, alignments))
        num_align.append(len(alignments))
    return np.array(is_found), np.array(num_align)


### Given a set of alignments, check if it contains a specific alignment (al1, al2).
### To save time, only look at the first 200 alignments in a set (since there can
### be hundreds of thousands in some cases...)
def is_alignment_found(al1, al2, alignments, max_align=200):
    """
    Test whether a specific alignment pair appears within a set of alignments.

    Parameters
    ----------
    al1 : str
        First aligned sequence string (with gap characters).
    al2 : str
        Second aligned sequence string (with gap characters).
    alignments : iterable
        Lazy iterator of alignment objects (e.g. from Biopython).
    max_align : int, optional
        Maximum number of alignments to inspect before giving up.
        Default 200 (guards against exponential tie sets).

    Returns
    -------
    bool
        True if ``(al1, al2)`` is found within the first ``max_align``
        alignments, False otherwise.
    """
    for i, aln in enumerate(alignments):
        if i >= max_align:
            break
        if all([a == b for a, b in zip(aln, [al1, al2])]):
            return True
    return False


### Load results for algorithm that checks whether manual alignments
### are within the best-scoring algorithmic alignments
def alignment_results():
    """
    Augment the Savage DataFrame with per-pair alignment correctness statistics.

    Loads pre-computed parameter-sweep results from ``OP.load_results_savage``
    and attaches ``freq_correct`` (fraction of parameter settings for which the
    manual alignment was recovered), ``num_align`` (number of optimal alignments
    at the best parameter setting), and ``is_correct`` (boolean recovery flag)
    as new columns on the Savage DataFrame.

    Returns
    -------
    None
        Modifies and prints diagnostics; the augmented DataFrame is not
        returned (results are side-effected onto ``df`` locally).

    Notes
    -----
    Results are broadcast to both rows of each pair (since each pair occupies
    two rows in the Savage DataFrame) using ``np.vstack`` tiling before
    ravelling.
    """
    df = load_savage_df()
    dfr, freq_correct = OP.load_results_savage()
    # Each pair has two rows; tile the per-pair stats across both rows
    df['freq_correct'] = np.vstack([freq_correct]*2).T.ravel()
    print(f"Percentage of pairs that never align properly: {np.mean(df.freq_correct==0)}")
    num_align = np.load(dfr.loc[2869, 'path'])[1]
    df['num_align'] = np.vstack([num_align]*2).T.ravel()
    correct = np.load(dfr.loc[2869, 'path'])[0]
    df['is_correct'] = np.vstack([correct]*2).T.ravel()
#   np.mean(df.loc[(df.is_correct==1), 'num_align'] == 1)
#   Counter(df.loc[(df.is_correct==1)&(df['num_align'] == 1), 'num_gaps'])


def get_submat(df):
    """
    Build a substitution observation matrix from Savage's manual alignments.

    Iterates over all aligned sequence pairs, strips indels, and tallies
    matched and substituted pitch-class pairs to produce an observation
    dictionary and the corresponding substitution matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Savage dataset as returned by ``load_savage_df``, containing
        ``PairNo``, ``tchroma``, and ``seq_aligned`` columns.

    Returns
    -------
    obs : collections.defaultdict
        Raw observation counts keyed by ``(tchroma_i, tchroma_j)`` tuples,
        including identity pairs (matches on the diagonal).
    letters : list
        Ordered list of pitch-class labels as returned by
        ``SM.convert_observations_to_matrix``.
    mat : np.ndarray
        Observation count matrix with shape ``(len(letters), len(letters))``.

    Notes
    -----
    Pairs where the two aligned sequences have different lengths are skipped
    with a printed warning.  Indels are removed symmetrically: positions where
    either sequence has a gap are excluded from both before counting.
    """
    obs = defaultdict(float)
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

        # Remove indels: find non-gap positions in each sequence,
        # then mask out positions where the *other* sequence has a gap
        idx1 = np.where(al1 != '-')[0]
        idx2 = np.where(al2 != '-')[0]
        tc1 = tc1[al2[idx1] != '-']
        tc2 = tc2[al1[idx2] != '-']

        # Get substitutions
        sub_idx = np.where(tc1 != tc2)[0]
        for i in sub_idx:
            obs[(tc1[i], tc2[i])] += 1

        # Get aligned notes
        same_idx = np.where(tc1 == tc2)[0]
        for k, v in Counter(tc1[same_idx]).items():
            obs[(k, k)] += v
    letters, mat = SM.convert_observations_to_matrix(obs, True)

    return obs, letters, mat
