from multiprocessing import Pool
import pickle

from Bio.Align import PairwiseAligner
import numpy as np
try:
    import parasail
    PARASAIL_WORKS = True
except:
    PARASAIL_WORKS = False
import pandas as pd

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.io import seq_io


########################################################################################
### Parasail pairwise alignments (fast, but only produces one alignment)


def get_pairwise(s1, s2, match=6, mismatch=-4, gap_open=4, gap_extend=3, alg='local'):
    custom_matrix = parasail.matrix_create(protein_letters, match, mismatch)
    if alg == 'local':
        result = parasail.sw_trace(s1, s2, gap_open, gap_extend, custom_matrix)
    elif alg == 'global':
        result = parasail.nw_trace(s1, s2, gap_open, gap_extend, custom_matrix)
    return result


def get_pairwise_score(s1, s2, match=6, mismatch=-4, gap_open=4, gap_extend=3, alg='local'):
    result = get_pairwise(s1, s2, match=match, mismatch=mismatch,
                          gap_open=gap_open, gap_extend=gap_extend, alg=alg)
    return result.score


def get_pairwise_alignment(s1, s2, match=6, mismatch=-4, gap_open=4, gap_extend=3, alg='local'):
    result = get_pairwise(s1, s2, match=match, mismatch=mismatch,
                          gap_open=gap_open, gap_extend=gap_extend, alg=alg)
    return np.array([list(result.traceback.query), list(result.traceback.ref)])


def get_pairwise_pid(s1, s2, match=6, mismatch=-4, gap_open=4, gap_extend=3, alg='local'):
    a1, a2 = get_pairwise_alignment(s1, s2, match=match, mismatch=mismatch,
                                    gap_open=gap_open, gap_extend=gap_extend, alg=alg)
    return np.sum(a1 == a2) / ((len(s1) + len(s2)) / 2)


def get_pairwise_family(df, tune_id):
    setting_ids, tchroma = df.loc[df.tune_id==tune_id, ['setting_id', 'tchroma']].values.T
    letter_seq = [''.join(letters[tp]) for tp in tchroma]
    distances = [get_pairwise_pid(s1, s2) for i, s1 in enumerate(letter_seq[:-1]) for s2 in letter_seq[i+1:]]
    N = len(letter_seq)
    mat = np.zeros((N,N), float)
    i, j = np.triu_indices(N, 1)
    mat[i,j] = distances
    mat[j,i] = distances
    return mat




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


########################################################################################
### Create pairwise sequence alignments for datasets


### Take two sequences (and kwargs), run pairwise alignment, and save
def run_single_pairwise(path, s1, s2, redo=False, **kwargs):
    if path.exists() and not redo:
        alignment = np.load(path)
    else:
        alignment = get_pairwise_alignment(s1, s2, **kwargs)
        if PARASAIL_WORKS:
            alignment = get_pairwise_alignment(s1, s2, **kwargs)
        else:
            alignment_list = get_pairwise_nhits(s1, s2, alg='local')
            alignment = np.array([list(x) for x in list(alignment_list[0])])
        np.save(path, alignment)
    return alignment


### Input generator for getting sequence alignments
def input_generator(res, setting2seq, path_results):
    for i, s1, s2 in zip(res.index, res['query'], res['target']):
        yield path_results.joinpath(f"{i}.npy"), ''.join(setting2seq[s1]), ''.join(setting2seq[s2])


### Run all pairwise alignments for a mmseqs output dataframe (res)
def run_all_pairwise_res(res, setting2seq, path_results, mp=False):
    inputs = input_generator(res, setting2seq, path_results)
    if mp:
        with Pool(N_PROC) as pool:
            CHUNK = len(res) // (N_PROC + 1)
            pairwise_align = pool.starmap(run_single_pairwise, inputs, CHUNK)
    else:
        pairwise_align = [run_single_pairwise(*i) for i in inputs]
    return pairwise_align


### Run all pairwise alignments for a given dataset
### Only run on pairs with PID > 0.5 (from mmseqs search results)
def run_all_pairwise(dataset, ref='setting_id', redo=False, mp=False, min_pid=0.5):
    path_letters = PATH_BASE.joinpath("Results/PairwiseAlignments", f"{dataset}_letters.pkl")
#   path_oct = PATH_BASE.joinpath("Results/PairwiseAlignments", f"{dataset}_oct.pkl")
    if path_letters.exists() and not redo:
        pairwise_align = pickle.load(open(path_letters, 'rb'))
#       pairwise_oct = pickle.load(open(path_oct, 'rb'))

    else:
        # Set up path to output
        path_results = PATH_BASE.joinpath("Results/PairwiseAlignments", dataset)
        path_results.mkdir(parents=True, exist_ok=True)

        # Load data
        if dataset == 'thesession':
            df = load_tunes.load_thesession_data()[0]
            res = seq_io.load_mmseqs_pairwise_thesession(df)
        elif dataset == 'meertens':
            df = load_tunes.load_meertens_data()[0]
            res = seq_io.load_mmseqs_pairwise_meertens(df)
        elif dataset == 'bronson':
            df = load_tunes.load_bronson_data()
            res = seq_io.load_mmseqs_pairwise_bronson(df)

        # Remove identical sequences and divergent sequences
        res = res.loc[(res.fident>=min_pid)&(res.fident<1)]

        # Key to map setting_id to protein letter sequence
        setting2seq = {s: np.array(list(protein_letters))[tp] for s, tp in zip(df[ref], df.tchroma)}

        # Calculate pairwise alignment (local)
        pairwise_align = run_all_pairwise_res(res, setting2seq, path_results, mp=mp)

#       # Map alignments to tmidi format (recover octave information)
#       pairwise_oct = utils.pairwise_reverse_mapping(df, res, pairwise_align, ref)

        pickle.dump(pairwise_align, open(path_letters, 'wb'))
#       pickle.dump(pairwise_oct, open(path_oct, 'wb'))

    return pairwise_align, pairwise_oct



if __name__ == "__main__":
    # Pre-process all pairwise alignments
    dataset_list = ['thesession', 'meertens', 'bronson']
    ref_names = ['setting_id', 'ref', 'ref']
    for i in range(3):
        run_all_pairwise(dataset_list[i], ref_names[i], redo=False, mp=True)




