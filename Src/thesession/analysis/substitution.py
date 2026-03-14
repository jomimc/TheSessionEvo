from collections import Counter, defaultdict
from itertools import product
import json
from math import comb
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.alignment import onset as OA
from thesession.alignment import pairwise as seq_align
from thesession.io import seq_io
from thesession import utils



######################################################################
### Generating simple substitution matrices

### Fixed match and mismatch scores for all substitutions
def basic_submat_A(diag=6, off_diag=-1.5, nmax=12, noise=0):
    N = len(letters)
    basic = np.zeros((N, N)) + off_diag
    np.fill_diagonal(basic, diag)
    if noise > 0:
        # Add a teeny bit of noise
        basic = basic + (np.random.rand(basic.size) - 0.5).reshape(basic.shape) * noise
    return basic


### Mismatch scores depend on substitution distance (semitones)
def basic_submat_B(diag=6, off_diag=-0.5, nmax=12, noise=0):
    N = len(letters)
    basic = np.zeros((N, N))
    for i, j in product(range(N), range(N)):
        if (i < nmax) and (j < nmax):
            d = abs(i - j)
            basic[i,j] = min(d, abs(12 - d)) * off_diag
    np.fill_diagonal(basic, diag)
    if noise > 0:
        # Add a teeny bit of noise
        basic = basic + (np.random.rand(basic.size) - 0.5).reshape(basic.shape) * noise
    return basic


### Mismatch scores depend on substitution distance (semitones)
def empirical_submat(sub_dist_log_odds, diag=6, off_diag=-0.5, nmax=12, noise=0):
    N = len(letters)
    mat = np.zeros((N, N))
    for i, j in product(range(N), range(N)):
        if (i < nmax) and (j < nmax):
            d = abs(i - j)
            basic[i,j] = min(d, abs(12 - d)) * off_diag
    np.fill_diagonal(basic, diag)
    if noise > 0:
        # Add a teeny bit of noise
        basic = basic + (np.random.rand(basic.size) - 0.5).reshape(basic.shape) * noise
    return basic


### Write a substitution matrix in the correct format for mmseqs
def write_mmseqs_sub_mat(path, submat, nmax=12):
    background = np.zeros(len(letters))
    background[:nmax] = 1 / nmax
    background_txt = ' '.join(["# Background (precomputed optional):"] + [f"{p:7.5f}" for p in background])
    with open(path, 'w') as o:
        o.write("# Custom substitution matrix\n")
        o.write(background_txt + '\n')
        o.write("# Lambda     (precomputed optional): 0.34657\n")
        o.write(('  ' + ' '.join(f" {l}     " for l in letters)).rstrip())
        for row, l in zip(submat, letters):
            o.write('\n')
            o.write(' '.join([f"{l}"] + [f"{item:7.4f}" for item in row]))
    

### Generate many substitution matrices for optimization
def generate_all_sub_mat():
    path_base = PATH_BASE.joinpath("substitution_matrices")
    diag_arr = np.arange(2, 12, 2)

    ### First, generate type A, where off-diagonal is constant
    off_diag = np.arange(-4, 1, 1)
    for d in diag_arr:
        for od in off_diag:
            path = path_base.joinpath(f"A_{d}_{od}.out")
            submat = basic_submat_A(d, od)
            write_mmseqs_sub_mat(path, submat)

    ### Second, generate type B, where mismatch is distance dependent
    diag_arr = np.arange(1, 6, 1)
    off_diag = np.arange(-1, 0, 0.2)
    for d in diag_arr:
        for od in off_diag:
            path = path_base.joinpath(f"B_{d}_{od:4.1f}.out")
            submat = basic_submat_B(d, od)
            write_mmseqs_sub_mat(path, submat)



######################################################################
### Convert msa formats


### Convert msa from sequence letters (A, B, etc.)
### to chromatic letters (C, C#, etc.)
def convert_msa_letters(msa):
    return np.vectorize(letter_key.__getitem__)(np.array(msa))


### Convert msa from sequence letters (A, B, etc.)
### to transposed chroma values
def convert_msa_letters_chroma(msa):
    msa = np.vectorize(position_key.__getitem__)(np.array(msa)).astype(float)
    # Convert gaps (assigned a value of 100) to nan values
    msa[msa==100] = np.nan
    return msa


######################################################################
### Get substitution rates / matrices from MSA


### Count substitution rates from MSA
def count_substitutions_from_msa(msa, gap_max=0.3, observations=None):
    if msa.dtype == np.string_:
        gap = np.mean(msa == '-', axis=0)
        gap_fn = lambda x: x == '-'
    else:
        gap = np.mean(np.isnan(msa), axis=0)
        gap_fn = lambda x: np.isnan(x)

    msa = msa[:,gap < gap_max]
    if isinstance(observations, type(None)):
        observations = defaultdict(int)

    for row in msa.T:
        count = Counter(row)
        keys = [k for k in count.keys() if not gap_fn(k)]

        # Add all of the same-note 'substitutions'
        for k in keys:
            observations[(k,k)] += comb(count[v], 2)

        # Add the different-note 'substitutions'
        for i, k1 in enumerate(keys[:-1]):
            for k2 in keys[i+1:]:
                observations[(min(k1, k2), max(k1, k2))] += count[k1] * count[k2]
    return observations


### Count overall substitution rates from a list of MSAs
def count_subs_many_msa(msa_list, gap_max=0.3):
    observations = defaultdict(int)
    for msa in msa_list:
        observations = count_substitutions_from_msa(msa, gap_max, observations)
    return observations


### Convert substitution observations (dictionary of counts)
### to a substitution matrix
def convert_observations_to_matrix(observations, chroma=False):
    if chroma:
        letters = np.arange(12)
    else:
        letters = sorted(set([x for y in observations.keys() for x in y]))
    key = {l:i for i, l in enumerate(letters)}
    mat = np.zeros((len(key), len(key)), float)
    for (k1, k2), v in observations.items():
        if k1 == '-' or k2 == '-':
            continue
        if chroma:
            i, j = [int(k%12) for k in [k1, k2]]
            mat[i,j] += v
            mat[j,i] += v
        else:
            i, j = key[k1], key[k2]
            mat[i,j] = v
            mat[j,i] = v
    return letters, mat


######################################################################
### Get substitution rates / matrices from pairwise alignments

def count_subs_pairwise_str(al1, al2, obs=None):
    if isinstance(obs, type(None)):
        obs = defaultdict(int)
    for a, b in zip(al1, al2):
        if (a != '-') and (b != '-'):
            obs[(min(a, b), max(a, b))] += 1
    return obs


def count_subs_pairwise_float(al1, al2, obs=None):
    if isinstance(obs, type(None)):
        obs = defaultdict(int)
    for i in np.where(~np.isnan(al1) & ~np.isnan(al2))[0]:
        a, b = al1[i], al2[i]
        obs[(min(a, b), max(a, b))] += 1
    return obs


def count_subs_many_pairwise(alignment_list):
    obs = defaultdict(int)
    for (al1, al2) in alignment_list:
        obs = count_subs_pairwise(al1, al2, obs)
    return obs



######################################################################
### Get substitution rates as a function of substitution distance


### Get substituion rate as a function of substitution distance (semitones)
### MSA should have be in sequence letter format (A, B, etc.)
def get_substition_distance_letter(msa, gap_max=0.3, obs=None):
    if isinstance(obs, type(None)):
        msa = convert_msa_letters(msa)
        obs = count_substitutions_from_msa(msa, gap_max)

    dist_sub = defaultdict(int)
    for k, v in obs.items():
        if ('-' in k) or (k[0] == k[1]):
            continue
        dist = abs(position_key[k[0]] - position_key[k[1]])
        dist = min(dist, abs(12 - dist))
        dist_sub[dist] += v

    X = np.arange(1, 7)
    Y = np.array([dist_sub.get(x, 0) for x in X])
    Y = Y / Y.sum()
    return X, Y


### Get substituion rate as a function of substitution distance (semitones)
### MSA should have float dtype
def get_substition_distance(msa, gap_max=0.3, obs=None):
    if isinstance(obs, type(None)):
        obs = count_substitutions_from_msa(msa, gap_max)

    dist_sub = defaultdict(int)
    for k, v in obs.items():
        if ('-' in k) or (k[0] == k[1]):
            continue
        elif np.any(np.isnan(k)):
            continue
        dist = abs(int(k[0]) - int(k[1]))
        dist_sub[dist] += v

    
#   xmin, xmax = [fn(dist_sub.keys()) for fn in [min, max]]
    xmin, xmax = [1, 17]
    X = np.arange(xmin, xmax + 1, 1)
    Y = np.array([dist_sub.get(x, 0) for x in X])
    Y = Y / Y.sum()
    return X, Y


######################################################################
### Process observations


def obs_to_dist_and_mat(observations):
    dist = {}
    matrices = {}
    for k, obs in observations.items():
        dist[k] = get_substition_distance('', obs=obs)
        matrices[k] = convert_observations_to_matrix(obs)
    return dist, matrices


def obs_mat_to_log_odds(mat):
    # Convert counts of pairs to probabilities of pairs
    # (matrix is symmetric about the diagonal, so we only
    #  count one side, plus the diagonal)
    prob = mat / np.sum(mat[np.triu_indices(len(mat), 0)])

    # To get the base probabilities of each note, we need 
    # to count the diagonal twice, and everything else in the row once
    base_prob = np.diag(prob) + np.sum(prob, axis=0)

    # The expected substitution probabilities is the outer
    # product of the base probabilities 
    exp_prob = np.outer(base_prob, base_prob)

    # Double the off-diagonal elements, to reflect the symmetry
    # (A->G is the same as G->A)
    exp_prob *= (2 - np.eye(len(mat)))

#   # Get expected substitution probabilities from base probabilies
#   exp_prob = np.outer(np.diag(prob), np.diag(prob))
#   # Make off-diagonals equivalent
#   exp_prob[np.where(~np.eye(exp_prob.shape[0],dtype=bool))] *= 2

    # Get the log odds
    log_odds = np.log(prob) - np.log(exp_prob)
    log_odds[~np.isfinite(log_odds)] = np.nan
    return log_odds


######################################################################
### Count observations in each dataset


def get_observations_pairwise(dataset, df, res, ref='setting_id', mp=False):
    # Load all pairwise alignments
    setting2seq = {s: np.array(list(protein_letters))[tp] for s, tp in zip(df[ref], df.tchroma)}
    path_results = PATH_BASE.joinpath("Results/PairwiseAlignments", dataset)
    pairwise_align = seq_align.run_all_pairwise_res(res, setting2seq, path_results, mp=mp)

    # Set up dictionaries for mapping
    setting2tchroma = {s: tp for s, tp in zip(df[ref], df.tchroma)}
    setting2tchroma_oct = {s: tp for s, tp in zip(df[ref], df.tchroma_octave)}

    # Set up container for observations
    obs = defaultdict(int)
    
    # Iterate over alignments (in protein letter format)
    for i, (al1, al2) in zip(res.index, pairwise_align):
        # Get song references for the two sequences
        s1, s2 = res.loc[i, ['query', 'target']]

        # Map alignments to the transposed midi representation
        tmidi_al1 = utils.reverse_mapping(setting2tchroma[s1], setting2tchroma_oct[s1], al1)
        tmidi_al2 = utils.reverse_mapping(setting2tchroma[s2], setting2tchroma_oct[s2], al2)

        # Count observations 
        obs = count_subs_pairwise_float(tmidi_al1, tmidi_al2, obs)

    return obs


def get_observations_msa(dataset, df):
    # Load all multiple sequence alignments
    msa_list = seq_io.load_tune_families_msa(dataset, df)
    # Convert to transposed midi
    msa_list_oct = utils.tune_families_reverse_mapping(df, msa_list)
    # Count observations
    return count_subs_many_msa(msa_list_oct)


def get_observations_onset_alignment(dataset, df, data, min_pid=0.85):
    # Load onset alignments (already in tmidi format)
    onset_alignments = OA.get_tune_family_alignments(df, data)
    # Set up container for observations
    obs = defaultdict(int)
    for al1, al2 in onset_alignments:
        # Count observations 
        obs = count_subs_pairwise_float(al1, al2, obs)
    return obs 


def get_observations_dataset(dataset, alg='pairwise', min_PID=0.5, sep_types=False, mp=False):
    # Set up path to results
    path_results = PATH_BASE.joinpath("Results/SubstitutionCounts", dataset)
    path_results.mkdir(parents=True, exist_ok=True)

    # Choose algorthm
    if alg == 'pairwise':
        # Load data
        df = load_tunes.load_df_dataset(dataset)
        res = seq_io.load_mmseqs_results_dataset(dataset, df)
        ref = {'thesession':'setting_id', 'meertens':'ref', 'bronson':'ref'}[dataset]

        # Apply sequence identity cutoff
        res = res.loc[(res.fident>=min_PID)&(res.fident<1)]
        if not len(res):
            return

        # Filter by tune type, if specified
        if sep_types:
            type_list = df['type'].unique()
            key = {k:v for k, v in zip(df[ref], df['type'])}
            res['q_typ'] = res['query'].map(key)
            res['t_typ'] = res['target'].map(key)
            for typ in type_list:
                path = path_results.joinpath(f"{alg}_{min_PID}_{typ}.pkl")
                idx = (res['q_typ'] == typ) & (res['t_typ'] == type)
                if np.sum(idx):
                    obs = get_observations_pairwise(dataset, df, res.loc[idx], ref=ref, mp=mp)
                    pickle.dump(obs, open(path, 'wb'))
                
        else:
            path = path_results.joinpath(f"{alg}_{min_PID}_all.pkl")
            obs = get_observations_pairwise(dataset, df, res, ref=ref, mp=mp)
            pickle.dump(obs, open(path, 'wb'))

    elif alg == 'msa':
        df = load_tunes.load_df_dataset(dataset)
        obs = get_observations_msa(dataset, df)
        path = path_results.joinpath(f"{alg}_{min_PID}_all.pkl")
        pickle.dump(obs, open(path, 'wb'))

    elif alg == 'onset_alignment':
        df, data = load_tunes.load_data_dataset(dataset)
        obs = get_observations_onset_alignment(dataset, df, data)
        path = path_results.joinpath(f"{alg}_{min_PID}_all.pkl")
        pickle.dump(obs, open(path, 'wb'))



if __name__ == "__main__":

    get_observations_dataset('thesession', 'pairwise', 0.85, False, True)


