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
    aligner = PairwiseAligner()
    aligner.mode = alg
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = -gap_open
    aligner.extend_gap_score = -gap_extend

    alignments = aligner.align(target, query)
    return alignments




