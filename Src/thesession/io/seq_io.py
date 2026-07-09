from collections import defaultdict

from Bio import Seq, SeqRecord, SeqIO
import numpy as np
import pandas as pd

from thesession.config import PATH_BASE, PATH_MMSEQS
from thesession import utils

######################################################################
### Write to fasta


### Required format for writing fasta using SeqIO
def make_pitch_seqrecord(ID, seq, convert=True):
    """
    Wrap a pitch sequence in a Biopython ``SeqRecord`` for FASTA output.

    Parameters
    ----------
    ID : str or int
        Identifier for the sequence record (used as the FASTA header).
    seq : array-like or str
        Pitch-class sequence to encode.  If ``convert=True`` this should
        be an integer array of tchroma values (0–11); if ``convert=False``
        it is expected to already be a letter string.
    convert : bool, optional
        If ``True`` (default), pass ``seq`` through ``utils.tchroma2seq``
        to map integer pitch classes to the protein-letter alphabet before
        wrapping.

    Returns
    -------
    Bio.SeqRecord.SeqRecord
        A SeqRecord whose sequence is the letter-encoded pitch string and
        whose ``id`` attribute is ``str(ID)``.
    """
    if convert:
        seq = utils.tchroma2seq(seq)
    return SeqRecord.SeqRecord(Seq.Seq(seq), id=str(ID))


### Write fasta file, given SeqRecord (or list of SeqRecord objects)
def write_fasta(path, seqrecord):
    """
    Write one or more ``SeqRecord`` objects to a FASTA file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file path.
    seqrecord : Bio.SeqRecord.SeqRecord or list of Bio.SeqRecord.SeqRecord
        Record(s) to write.

    Returns
    -------
    None
    """
    SeqIO.write(seqrecord, path, "fasta")


### Write many sequences to one fasta file
def write_all_seq_to_fasta(seq_list, id_list, path, convert=True):
    """
    Convert a collection of pitch sequences to FASTA and write to disk.

    Parameters
    ----------
    seq_list : iterable of array-like or str
        Pitch sequences in the same order as ``id_list``.
    id_list : iterable of str or int
        Identifiers, one per sequence.
    path : str or pathlib.Path
        Destination file path.
    convert : bool, optional
        Passed to ``make_pitch_seqrecord``.  If ``True`` (default),
        integer tchroma arrays are converted to letter strings.

    Returns
    -------
    None
    """
    records = [make_pitch_seqrecord(ID, seq, convert) for ID, seq in zip(id_list, seq_list)]
    write_fasta(path, records)


def write_parts_thesession(parts):
    """
    Write all TheSession parts to a single FASTA file for MMseqs2 input.

    The output file is created at
    ``PATH_MMSEQS / 'thesession_parts/all_seq_thesession_parts.fasta'``,
    with parent directories created if necessary.  Each record's sequence
    is the pitch-class (``tmidi % 12``) letter representation of the part.

    Parameters
    ----------
    parts : dict
        Mapping from part_id (str) to ``(part, nbars)`` where
        ``part = (dur_array, tmidi_array)``.

    Returns
    -------
    None
    """
    path = PATH_MMSEQS.joinpath('thesession_parts/all_seq_thesession_parts.fasta')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reduce absolute MIDI to pitch class before encoding as letters
    records = [make_pitch_seqrecord(ID, part[0][1] % 12)  for ID, part in parts.items()]
    write_fasta(path, records)



######################################################################
### Load mmseqs results


def load_mmseqs_pairwise(df, dataset, annotate=True):
    """
    Load an MMseqs2 result file and optionally annotate hits with
    within-family labels.

    The function reads the tab-separated ``result.m8`` file, removes
    self-hits and deduplicated reverse pairs (keeping only one of
    (A, B) and (B, A)), then optionally adds an ``in_fam`` boolean
    column indicating whether query and target belong to the same tune
    family.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table used to look up family membership.  Required
        columns depend on ``dataset`` (see ``get_families_key``).
    dataset : str
        Name of the dataset subdirectory under ``PATH_MMSEQS``.  Also
        determines which metadata columns are used for family grouping.
        Supported values: ``'thesession_parts'``, ``'thesession_tunes'``,
        ``'meertens'``, ``'savage_english'``.
    annotate : bool, optional
        If ``True`` (default), add the ``in_fam`` column by calling
        ``annotate_alignment``.

    Returns
    -------
    res : pandas.DataFrame
        Filtered (and optionally annotated) MMseqs2 hit table.

    Notes
    -----
    Deduplication is performed by sorting each (query, target) pair
    alphabetically and then dropping rows where that sorted pair has
    been seen before.  This ensures each unordered pair is represented
    exactly once.
    """
    path = PATH_BASE.joinpath(f'MMseqs/{dataset}/result.m8')
    res = pd.read_csv(path, sep='\t')

    # Remove self-comparisons
    res = res.loc[res['query'] != res['target']]

    # Remove duplicates
    # Sort each pair so that (A,B) and (B,A) produce the same key, then deduplicate
    pairs = res[['query', 'target']].values
    df_sort = pd.DataFrame(np.sort(pairs, axis=1), columns=['a', 'b'])
    res = res.loc[df_sort.duplicated().values==False]

    if annotate:
        # Check whether hits are in the same tune family
        families, family_key = get_families_key(df, dataset)
        res = annotate_alignment(res, families, family_key)
    return res


def get_families_key(df, dataset):
    """
    Build a family-membership lookup from a metadata DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table for the given dataset.
    dataset : str
        One of ``'thesession_parts'``, ``'thesession_tunes'``,
        ``'meertens'``, or ``'savage_english'``.  Determines which
        columns are used as the family identifier and the sequence
        identifier.

    Returns
    -------
    families : collections.defaultdict of set
        Maps each family identifier to the set of sequence identifiers
        (e.g. setting IDs) that belong to it.
    family_key : dict
        Maps each sequence identifier back to its family identifier,
        for O(1) lookup when annotating hit pairs.
    """
    x, y = {'thesession_parts': ('tune_id', 'setting_id'),
            'thesession_tunes': ('tune_id', 'setting_id'),
            'meertens': ('song_id', 'ref'),
            'savage_english': ('chapter', 'ref')
            }[dataset]
    families = defaultdict(set)
    family_key = {}
    for t, s in zip(df[x], df[y]):
        families[t].add(s)
        family_key[s] = t
    return families, family_key


def annotate_alignment(res, families, family_key):
    """
    Add an ``in_fam`` column to a hit table indicating within-family pairs.

    Parameters
    ----------
    res : pandas.DataFrame
        MMseqs2 hit table with at least the columns ``query`` and
        ``target``.
    families : collections.defaultdict of set
        Mapping from family identifier to the set of member sequence IDs,
        as returned by ``get_families_key``.
    family_key : dict
        Mapping from each sequence ID to its family identifier.

    Returns
    -------
    res : pandas.DataFrame
        The same DataFrame with a new boolean column ``in_fam`` that is
        ``True`` when query and target belong to the same family.
    """
    res['in_fam'] = [t in families[family_key[q]] for q, t in zip(res["query"], res["target"])]
    return res
