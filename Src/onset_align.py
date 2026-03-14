from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from global_variables import *
import plots
import seq_align
import utils


def align_melodies(onset1, onset2, tchroma1, tchroma2, seq_align):
    pass


### Given two sequences of the same length,
### find the longest matching segment
def find_seq_matches(s1, s2):
    if isinstance(s1, str):
        s1 = np.array(list(s1))
        s2 = np.array(list(s2))
    is_equal = s1 == s2
    idx = np.where(is_equal)[0]
    is_connected = np.diff(idx) == 1
    clusters = []
    clust = [idx[0]]
    for i in range(len(idx) - 1):
        if is_connected[i]:
            clust.append(idx[i+1])
        else:
            clusters.append(clust)
            clust = [idx[i+1]]
    return clusters


def find_longest_match(s1, s2):
    clusters = find_seq_matches(s1, s2)
    clust_len = np.array([len(c) for c in clusters])
    return clusters[np.argmax(clust_len)]


### Takes two melodies, with tchroma and bar_onset,
### gets an alignment, finds the longest ungapped part,
### and checks that the bar onsets are the same
def check_match(tune1, tune2):
    lseq1 = ''.join(letters[tune1['tchroma']])
    lseq2 = ''.join(letters[tune2['tchroma']])
    s1, s2 = np.array(list(seq_align.get_pairwise_alignment(lseq1, lseq2)))
    aligned_onset1 = utils.reverse_mapping(tune1['tchroma'], tune1['bar_onset'], s1)
    aligned_onset2 = utils.reverse_mapping(tune2['tchroma'], tune2['bar_onset'], s2)
    match_idx = find_longest_match(s1, s2)
    return np.all(aligned_onset1[match_idx] == aligned_onset2[match_idx])


### Takes as tunes the dict output of music21 parser,
### alongside the tchroma sequences, and optionally precomputed alignments (s1, s2)
def match_seq_align_to_bar_onsets(tune1, tune2, s1=[], s2=[], ncheck=5, plot=True):
    if len(s1) == 0:
        lseq1 = ''.join(letters[tune1['tchroma']])
        lseq2 = ''.join(letters[tune2['tchroma']])
        s1, s2 = np.array(list(seq_align.get_pairwise_alignment(lseq1, lseq2)))

    # Take the longest matches
    clusters = find_seq_matches(s1, s2)
    idx = np.argsort([len(c) for c in clusters])[::-1][:ncheck]
    score = []
    for i in idx:
        on1, on2 = shift_onset(tune1, tune2, s1, s2, clusters[i])
        score.append(check_matches_shifted_onset(on1, on2, tune1['tchroma'], tune2['tchroma']))
    on1, on2 = shift_onset(tune1, tune2, s1, s2, clusters[idx[np.argmax(score)]])
    if plot:
        plots.plot_aligned_notes(on1, on2, tune1, tune2)
    return on1, on2


def shift_onset(tune1, tune2, s1, s2, match_idx):
    start1, end1 = utils.reverse_mapping_idx(tune1['tchroma'], s1)
    start2, end2 = utils.reverse_mapping_idx(tune2['tchroma'], s2)
    match_loc1 = sum(x!='-' for x in s1[:match_idx[0]])
    match_loc2 = sum(x!='-' for x in s2[:match_idx[0]])
    shifted_onset1 = np.array(tune1['onsets']) - tune1['onsets'][start1+match_loc1]
    shifted_onset2 = np.array(tune2['onsets']) - tune2['onsets'][start2+match_loc2]
    return shifted_onset1, shifted_onset2
    

def check_matches_shifted_onset(on1, on2, tchroma1, tchroma2):
    on1 = np.round(on1, 2)
    on2 = np.round(on2, 2)
    idx1 = np.in1d(on1, on2)
    idx2 = np.in1d(on2, on1)
    return np.sum(tchroma1[idx1] == tchroma2[idx2])


### Match all possible bar onsets, and check how many notes align
def check_all_measure_onsets(tune1, tune2, plot=False):
    # Get note onsets
    on1 = np.array(tune1['onsets'])
    on2 = np.array(tune2['onsets'])
    # Find where notes fall on the start of a measure
    idx1 = np.where(np.array(tune1['bar_onset']) == 0)[0]
    idx2 = np.where(np.array(tune2['bar_onset']) == 0)[0]
    # Calculate all unique offsets (i.e. ignore duplicates which end up with the same alignment)
    offsets = sorted(set(x-y for x, y in product(on1[idx1], on2[idx2])))
    score = []
    for o in offsets:
        score.append(check_matches_shifted_onset(on1 - o, on2, tune1['tchroma'], tune2['tchroma']))
    on1 = on1 - offsets[np.argmax(score)]
    if plot:
        plots.plot_aligned_notes(on1, on2, tune1, tune2)
    return on1, on2


def onsets_to_alignment(on1, on2, tune1, tune2, prune=True):
    on1 = np.round(on1, 2)
    on2 = np.round(on2, 2)

    align_on = np.unique(sorted(set(on1).union(on2)))
    idx1 = np.in1d(align_on, on1)
    idx2 = np.in1d(align_on, on2)

    al1 = np.zeros(align_on.size) * np.nan
    al2 = np.zeros(align_on.size) * np.nan

    al1[idx1] = tune1['tmidi']
    al2[idx2] = tune2['tmidi']

    if prune:
        idx = prune_edges_nan(al1, al2)
        return align_on[idx], al1[idx], al2[idx]
    
    return align_on, al1, al2


### Get a waterman-smith type local alignment
def prune_edges_nan(al1, al2):
    idx = np.ones(al1.size, bool)
    isnan1 = np.isnan(al1)
    isnan2 = np.isnan(al2)
    for isnan in [isnan1, isnan2]:
        for i in range(len(al1)):
            if not isnan[i]:
                break
            else:
                idx[i] = False

        for i in range(len(al1)):
            if not isnan[-1-i]:
                break
            else:
                idx[-1-i] = False
    return idx


def align_tunes_by_measure_onsets(tune1, tune2):
    on1, on2 = check_all_measure_onsets(tune1, tune2)
    align_on, al1, al2 = onsets_to_alignment(on1, on2, tune1, tune2)
    return align_on, np.array([al1, al2])


def align_all_score_list(tune_list):
    ngaps, nmatches, nmismatches = [], [], []
    len_gap_mismatch = []
    alignments = []
    mismatch_distance = []
    for i in range(len(tune_list) - 1):
        for j in range(i + 1, len(tune_list)):
            align_on, (al1, al2) = align_tunes_by_measure_onsets(tune_list[i], tune_list[j])
            alignments.append([align_on, al1, al2])

            ngaps.append(np.sum(np.isnan(al1)) + np.sum(np.isnan(al2)))
            nmatches.append( np.sum(al1 == al2))

            nongap = (~np.isnan(al1)) & (~np.isnan(al2))
            idx_mismatch = al1[nongap] != al2[nongap]
            nmismatches.append(np.sum(idx_mismatch))
            mismatch_distance.append(np.abs(al1[nongap][idx_mismatch] - al2[nongap][idx_mismatch]))

            len_gap_mismatch.append(count_length_nonmatch(al1, al2))
    ngaps, nmatches, nmismatches = [np.array(x) for x in [ngaps, nmatches, nmismatches]]
    return alignments, ngaps, nmatches, nmismatches, len_gap_mismatch, mismatch_distance


def find_nonmatching_idx(al1, al2):
    not_equal = al1 != al2
    idx = np.where(not_equal)[0]
    if len(idx) == 0:
        return np.empty(0)
    is_connected = np.diff(idx) == 1
    clusters = []
    clust = [idx[0]]
    for i in range(len(idx) - 1):
        if is_connected[i]:
            clust.append(idx[i+1])
        else:
            clusters.append(clust)
            clust = [idx[i+1]]
    return clusters

def count_length_nonmatch(al1, al2):
    clusters = find_nonmatching_idx(al1, al2)
    len_mismatch = []
    for c in clusters:
        len_mismatch.append([np.sum(~np.isnan(al1[c])),
                             np.sum(~np.isnan(al2[c]))])
    return np.array(len_mismatch)


def get_tune_family_sub_dist(df, data, minsize=5):
    setting_idx = {s:i for i, s in zip(df.index, df.setting_id)}
    pid_thresholds = [0.5, 0.75, 0.85, 0.95]
    # Find tune families that have at least 5 tunes 
    tune_ids, counts = np.array([[k, v] for k, v in df.tune_id.value_counts().items() if v >= 5]).T
    sub_dist = {tid:{pid:[] for pid in pid_thresholds} for tid in tune_ids}
    for tid in tqdm(tune_ids):
        print(tid)
        setting_ids = df.loc[df.tune_id == tid, 'setting_id'].values
        tunes = [data[s] for s in setting_ids]
        # Align tunes
        alignments, ngaps, nmatches, nmismatches, len_gap_mismatch, mismatch_distance = align_all_score_list(tunes)
        # Calculate percent identity
        pid = nmatches / np.array([x[0].size for x in alignments])
        for threshold in pid_thresholds:
            # Find similar tunes, according to pid threshold
            idx = np.where((pid >= threshold) & (pid < 1))[0]
            for i in idx:
                sub_dist[tid][threshold].extend([d for d in mismatch_distance[i]])
    return sub_dist


def get_tune_family_alignments(df, data, minsize=5):
    # Find tune families that have at least 5 tunes 
    tune_ids, counts = np.array([[k, v] for k, v in df.tune_id.value_counts().items() if v >= 5]).T
    for tid in tqdm(tune_ids):
        print(tid)
        setting_ids = df.loc[df.tune_id == tid, 'setting_id'].values
        alignments = []
        tunes = [data[s] for s in setting_ids]
        for i in range(len(tunes) - 1):
            for j in range(i + 1, len(tunes)):
                alignments.append(align_tunes_by_measure_onsets(tunes[i], tunes[j])[1])
        # Calculate percent identity
        pid = nmatches / np.array([x[0].size for x in alignments])
        for threshold in pid_thresholds:
            # Find similar tunes, according to pid threshold
            idx = np.where((pid >= threshold) & (pid < 1))[0]
            for i in idx:
                sub_dist[tid][threshold].extend([d for d in mismatch_distance[i]])
    return sub_dist


def check_seq_align_onsets(tune1, tune2):
    lseq1 = ''.join(letters[tune1['tchroma']])
    lseq2 = ''.join(letters[tune2['tchroma']])
    s1, s2 = np.array(list(seq_align.get_pairwise_alignment(lseq1, lseq2)))
    aligned_onset1 = utils.reverse_mapping(tune1['tchroma'], tune1['bar_onset'], s1)
    aligned_onset2 = utils.reverse_mapping(tune2['tchroma'], tune2['bar_onset'], s2)
    return aligned_onset1, aligned_onset2


### Does not work yet. Trying on pair 6...
def onsets_to_alignment_fuzzy(on1, on2, tune1, tune2, prune=True):
    on1 = np.round(on1, 2)
    on2 = np.round(on2, 2)

    # These are the onset matches
    idx1 = np.in1d(on1, on2)
    idx2 = np.in1d(on2, on1)

    onset1 = on1[~idx1]
    onset2 = on2[~idx2]

    # These are the remaining onsets that have close matches 
    i1, i2 = match_offset_onsets(onset1, onset2).T
    
    # This is the total set of matches
    idx1 = np.array(sorted(list(np.where(idx1)[0]) + list(np.where(~idx1)[0][i1])))
    idx2 = np.array(sorted(list(np.where(idx2)[0]) + list(np.where(~idx2)[0][i2])))

    # This is the size of the alignment
    len_align = idx1.size + (len(onset1) - len(i1)) + (len(onset2) - len(i2))
    al1 = np.zeros(len_align) * np.nan
    al2 = np.zeros(len_align) * np.nan

    # What are the indices of the alignment?
    skip = 0
    for i, o in enumerate(on1):
        if i in idx1:
            al1[i + skip] = o
        else:
            skip += 1

    skip = 0
    for i, o in enumerate(on2):
        if i in idx1:
            al2[i + skip] = o
        else:
            skip += 1

    return al1, al2


def match_offset_onsets(on1, on2, max_offset=0.5):
    # Create cost matrix (absolute distance between elements)
    cost_matrix = np.abs(on1[:, None] - on2[None, :])

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Return list of (index_in_on1, index_in_on2, distance)
    matches = np.array([[i, j] for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < max_offset])
    return matches



