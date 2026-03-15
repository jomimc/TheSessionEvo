from multiprocessing import Pool
import pickle

from Bio.Align import PairwiseAligner
import numpy as np
import pandas as pd

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.io import seq_io


########################################################################################
### Biopython pairwise alignments (slow, but offers more control)


def get_pairwise_nhits(target, query, match=6, mismatch=-4, gap_open=4, gap_extend=3, alg='global'):
    """
    Compute all optimal pairwise alignments between two sequences using Biopython.

    Parameters
    ----------
    target : str or sequence
        The target (reference) sequence to align against.
    query : str or sequence
        The query sequence to align.
    match : int or float, optional
        Score awarded for a matched position. Default 6.
    mismatch : int or float, optional
        Score awarded for a mismatched position (typically negative). Default -4.
    gap_open : int or float, optional
        Penalty for opening a gap (applied as a cost, so stored as negative
        internally). Default 4.
    gap_extend : int or float, optional
        Penalty for extending a gap by one position. Default 3.
    alg : str, optional
        Alignment mode: ``'global'`` (Needleman-Wunsch) or ``'local'``
        (Smith-Waterman). Default ``'global'``.

    Returns
    -------
    alignments : Bio.Align.PairwiseAlignments
        Lazy iterator of all optimal alignments.  Call ``len(alignments)`` to
        count them or iterate to inspect individual alignment objects.

    Notes
    -----
    Gap penalties are passed as positive numbers but stored as negatives
    internally by Biopython (open_gap_score = -gap_open). This function is
    slower than MMseqs2 but provides exact alignment with full control over
    scoring parameters, making it useful for parameter optimisation on small
    datasets such as the Savage corpus.
    """
    aligner = PairwiseAligner()
    aligner.mode = alg
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    # Biopython expects negative gap scores; negate the positive penalty inputs
    aligner.open_gap_score = -gap_open
    aligner.extend_gap_score = -gap_extend

    alignments = aligner.align(target, query)
    return alignments




