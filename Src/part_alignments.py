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

from global_variables import *
import utils


#######################################################
### Aligning parts identified using mmseqs

### Check that total durations add up
def compare_part_duration(part1, part2):
    return np.sum(part1[0]) == np.sum(part2[0])


### Compare two parts by comparing pitches aligned on a grid
def compare_parts(part1, part2):
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
    if isinstance(factor, type(None)):
        factor = utils.get_common_denominator([part1[0], part2[0]])
    tc1 = utils.get_tchroma_grid(part1[1], part1[0], factor)
    tc2 = utils.get_tchroma_grid(part2[1], part2[0], factor)
    return tc1, tc2, factor


### Check if parts differ by an octave,
### and if so correct by changing the pitch of one part
def correct_octave_diff(tc1, tc2, threshold=8):
    m1 = np.mean(tc1)
    m2 = np.mean(tc2)
    diff = m1 - m2
    if abs(diff) > threshold:
        return tc1 - 12 * np.round(diff / 12), tc2
    return tc1, tc2


### Align and compare two parts
def analyse_part_alignment(part1, part2, meter):
    # Put tunes on a grid
    tc1, tc2, factor = part2grid(part1, part2)

    # Correct for potential octave difference
    tc1, tc2 = correct_octave_diff(tc1, tc2)

    # Check this number! It might be off by a factor of two (or else I have already corrected it!)
    grid_per_bar = int(4 * eval(meter) * factor)

    idx = tc1 != tc2

    onset = np.arange(len(tc1))
    bar_onset = onset % grid_per_bar
    bar = onset // grid_per_bar

    out = {'sub_bar': bar[idx],
           'sub_pos': bar_onset[idx] / grid_per_bar,
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
    # Create the graph of identical parts
    G = nx.Graph()

    # Get part lengths
    uniq_parts = np.unique(np.concatenate([res['query'].unique(), res['target'].unique()]))
    part_length = {p: len(parts[p][0][0]) for p in uniq_parts}
    res['qlen'] = res['query'].map(part_length)
    res['tlen'] = res['target'].map(part_length)

    # Identical tunes must not only have "fident" ("fraction identity") = 1,
    # but the alignment length must also be the same size as both original sequences
    same_len = (res.alnlen==res['qlen']) & (res.alnlen==res['tlen'])
    identical_parts = res.loc[(res.fident==1)&(same_len), ['query', 'target']].values
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

    return res.loc[(~res["query"].isin(to_remove))&(~res["target"].isin(to_remove))]


### Filter and annotate pairs of parts identified using mmseqs
def annotate_res(df, df_parts, res, parts, redo=False):
    path_results = [PATH_BASE.joinpath("Results", n) for n in ["pairs_thesession_parts.pkl",
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
    cols = ['setting_id', 'tune_id', 'part_no', 'num_parts'] 
    multikey = defaultdict(dict)
    for p, vals in zip(df_parts.part_id, df_parts[cols].values):
        for c, v in zip(cols, vals):
            multikey[c][p] = v

    col2 = ['setting', 'tune', 'part', 'num_parts']
    for a in ['query', 'target']:
        for c1, c2 in zip(cols, col2):
            c3 = f"{a}_{c2}"
            res[c3] = res[a].map(multikey[c1])

    # Annotate in_fam
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
    out = np.array([compare_parts(parts[i][0], parts[j][0]) for i, j in zip(res['query'], res['target'])])
    res['eq_dur'] = out[:,0].astype(bool)
    res['frac_eq'] = out[:,1].astype(float)

    # Reduce to true hits
    res0 = res.loc[(res.eq_dur)&(res.eq_meter)&(res.frac_eq>0.5)&(res.frac_eq<1)]

    # Annotate duration and discretization details
    res0['total_dur'] = res0['query'].apply(lambda x: np.sum(parts[x][0][0])) # Total duration in eigth note units
    res0['factor'] = [utils.get_common_denominator([parts[q][0][0], parts[t][0][0]])
                      for q, t in zip(res0['query'], res0['target'])] # Factor used to map lowest duration to grid spacing
    res0["grid_per_bar"] = [int(4 * eval(m) * f) for m, f in zip(res0.target_meter, res0.factor)] # Number of grid points per bar
    res0['nbars'] = res0['query'].apply(lambda q: parts[q][1]) # Number of bars in part

    # Remove songs where meter in ABC is not the same as in the header
    res0 = res0.loc[(res0['total_dur'] / res0['nbars']) == (4 * res0.target_meter.apply(eval))]

    # Check that the meter is correct (sometimes header annotation is wrong)
    res0['correct_meter'] = (res0['total_dur'] / res0['nbars']) == (4 * res0.target_meter.apply(eval))

    # Annotate mode
    res0['target_mode'] = [utils.check_mode(parts[t][0][1] % 12) for t in res0['target']]
    res0['query_mode'] = [utils.check_mode(parts[t][0][1] % 12) for t in res0['query']]

    # Annotate tune counts
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
    obs = defaultdict(float)
    weights = utils.inverse_frequency_weights(res, alpha)
    for subs, m, g, w in zip(mismatches.sub_notes, res.target_meter, res.grid_per_bar, weights):
        unit = eval(m) / g * w
        for a, b in zip(*subs.T):
            obs[(a%12,b%12)] += unit
    
    for matches, m, g, w in zip(mismatches.match_notes, res.target_meter, res.grid_per_bar, weights):
        unit = eval(m) / g * w
        for k, v in matches.items():
            obs[(k%12,k%12)] += unit * v

    return obs
    

### Get the melodic interval distribution from tunes
def get_mint_dist(tunes):
    X = np.arange(1, 18)
    count = {x: 0 for x in X}
    for d in tunes.values():
        C = Counter(np.abs(np.diff(d['tmidi'])))
        for k in count.keys():
            count[k] += C.get(k, 0)
    Y = np.array([count.get(x, 0) for x in X])
    Y = Y / Y.sum()
    return X, Y


### Get the melodic interval distribution from bars
def get_mint_dist_bars(bars):
    X = np.arange(1, 18)
    count = {x: 0 for x in X}
    for d in bars.values():
        C = Counter(np.abs(np.diff(d[1])))
        for k in count.keys():
            count[k] += C.get(k, 0)
    Y = np.array([count.get(x, 0) for x in X])
    Y = Y / Y.sum()
    return X, Y


def get_mint_dist_part(part):
    X = np.arange(1, 18)
    count = {x: 0 for x in X}
    for d in part.values():
        C = Counter(np.abs(np.diff(d[1])))
        for k in count.keys():
            count[k] += C.get(k, 0)
    Y = np.array([count.get(x, 0) for x in X])
    Y = Y / Y.sum()
    return X, Y


#######################################################
### Multiple part alignment


def get_msa(res, parts, query, min_pid=0.85, max_grid=16, nbars=8, factor=2, part_list=None):
    if isinstance(part_list, type(None)):
        part_list = np.unique(res.loc[((res['query']==query)|(res['target']==query))
                                      &(res.fident>=min_pid)&(res.target_meter==res.query_meter),
                                      ['query', 'target']].values)
#   factor = max_grid / (4 * eval(meter))
    ngrid = nbars * max_grid

    tc1, tc2 = part2grid(parts[part_list[0]][0], parts[part_list[1]][0], factor)[:2]
    tc1, tc2 = correct_octave_diff(tc1, tc2)
    tc_list = [tc1[:ngrid], tc2[:ngrid]]
    for i in range(2, len(part_list)):
        tc1, tc3 = part2grid(parts[part_list[0]][0], parts[part_list[i]][0], factor)[:2]
        tc1, tc3 = correct_octave_diff(tc1, tc3)
        if tc3.size >= ngrid:
            tc_list.append(tc3[:ngrid])
    return np.array(tc_list)


def get_msa_family(res, parts, tune_id, p=0, min_pid=0.85, max_grid=16, nbars=8, factor=2, part_list=None):
    if isinstance(part_list, type(None)):
        part_list = np.unique(res.loc[(res['query_tune']==tune_id)&(res['target_tune']==tune_id)
                                      &(res.fident>=min_pid)&(res.target_meter==res.query_meter)
                                      &(res['query_part']==p)&(res['target_part']==p),
                                      ['query', 'target']].values)

    if len(part_list) == 0:
        return [], []

    ngrid = nbars * max_grid

    tc1, tc2 = part2grid(parts[part_list[0]][0], parts[part_list[1]][0], factor)[:2]
    tc1, tc2 = correct_octave_diff(tc1, tc2)
    tc_list = [tc1[:ngrid], tc2[:ngrid]]
    idx_out = [0,1]
    for i in range(2, len(part_list)):
        tc1, tc3 = part2grid(parts[part_list[0]][0], parts[part_list[i]][0], factor)[:2]
        tc1, tc3 = correct_octave_diff(tc1, tc3)
        if tc3.size >= ngrid:
            tc_list.append(tc3[:ngrid])
            idx_out.append(i)
    return part_list[idx_out], np.array(tc_list)


def get_position_conservation(msa):
    N = msa.shape[1]
    H = np.zeros(N, float)
    for i in range(N):
        count = np.array(list(Counter(msa[:,i]).values()))
        if count.size == 1:
            H[i] = 0
        else:
            H[i] = entropy(count / count.sum())
    return H


def count_ngrams(parts, part_list, factor=2, context=4):
    count = np.zeros([12]*context, float)
    for p in tqdm(part_list):
        dur, midi = parts[p][0]
        if np.any(dur < 1 / factor):
            continue
        tc = np.array(utils.get_tchroma_grid(midi, dur, factor) % 12, int)
        for i in range(len(tc) - context + 1):
            count[tuple(tc[i:i+context])] += 1
    return count


def get_novelty_profile(res, parts, q, N=4):
    test = np.unique(res.loc[(res['query']==q)|(res['target']==q), ['query', 'target']].values)
    parts_keys = np.array(list(parts.keys()))
    not_in_test = np.array(list(set(parts_keys).difference(test)))
    count = count_ngrams(parts, not_in_test, 2, 4)

    msa = get_msa(res, parts, q, 0.5, 8)
    msa_tc = np.array(msa % 12, int)
    msa_prev = np.array([[count[tuple(msa_tc[i,j:j+N])] for j in range(msa.shape[1] - N + 1)] for i in range(msa.shape[0])])
    return msa_prev








