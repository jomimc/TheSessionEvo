from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pearsonr, multinomial, entropy

from thesession.config import *
from thesession.alignment import parts as PA
from thesession import utils


###################################################################
### Key/mode statistics and modal profiles


def get_key_mode_indices(target_mode):
    modes = {m:i for i, m in enumerate(list(MODES.keys()))}
    if '#' in target_mode:
        key_idx = chromatic_map[target_mode[:2]]
        mode_idx = modes[target_mode[2:]]
    else:
        key_idx = chromatic_map[target_mode[0]]
        mode_idx = modes[target_mode[1:]]
    return (key_idx, mode_idx)


def get_key_mode_priors(df, nkey=12, nmode=4):
    key_mode_indices = df['mode'].apply(get_key_mode_indices).values
    count = Counter(key_mode_indices)
    prob = np.zeros((nkey, nmode), float)
    for (i, j), v in count.items():
        prob[i,j] = v
    prob = prob / np.sum(prob)
    return prob


### Count tonal hierarchies for each mode
def get_modal_profiles(df, data):
    modes = list(MODES.keys())

    # First prune dataset:
    #   No ambiguous modes (pentatonic, mixed, or indeterminate)
    df = df.loc[df['inferred_mode'].isin(modes)]

    #   Make sure my simple algorithm annotations match the dataset annotations
    df = df.loc[df['inferred_mode'] == df['mode'].apply(lambda x: x[1:])]

    #   Exclude any tunes with key changes
    df = df.loc[df.has_key_change == False]

    # Set up mode profiles
    profiles = {}

    # Count pitch histograms
    bins = np.arange(-0.5, 12, 1)
    for mode in modes: 
        hist = []
        for i in df.loc[df['inferred_mode']==mode, 'setting_id']:
            h = np.histogram(data[i]['tchroma'], bins=bins)[0]
            hist.append(h / h.sum()) 
        profiles[mode] = np.mean(hist, axis=0)
    return profiles


### Compute likelihood / correlations by comparing a tune's tonal hierarchy
### against all possible keys and modes 
def profile_correlation(tchroma, mode_profiles, alg='bayesian'):
#   if alg == 'bayesian':
#       tchroma = tchroma.astype(int)
#   elif alg == 'pearson':
    tchroma = tchroma.astype(int) % 12
    bins = np.arange(-0.5, 12, 1)
    hist = np.histogram(tchroma, bins=bins)[0]
    N = len(tchroma)

    key_idx = np.arange(12)
    modes = list(MODES.keys())
    score = np.zeros((len(key_idx), len(modes)), float)
    for i in key_idx:
        for j, m in enumerate(modes):
            if alg == 'bayesian':
#               score[i,j] = np.mean(np.log(mode_profiles[m][(tchroma - i) % 12]))
                score[i,j] = np.log(multinomial.pmf(np.roll(hist, -i), N, mode_profiles[m]))#/ N**0.5
            elif alg == 'pearson':
                score[i,j] = pearsonr(mode_profiles[m], np.roll(hist, -i))[0]
    return score


### Compute the most likely key and mode for a tune
def assign_key_and_mode(tchroma, mode_profiles, alg='bayesian', priors=None):
    modes = list(MODES.keys())
    score = profile_correlation(tchroma, mode_profiles, alg=alg)
    if not isinstance(priors, type(None)):
        score += np.log(priors)
    i, j = np.unravel_index(np.nanargmax(score), score.shape)
    return chromatic_notes[i] + modes[j]


def compute_tonal_ambiguity(tchroma, mode_profiles, priors=None):
    score = profile_correlation(tchroma, mode_profiles, alg='bayesian')
    if not isinstance(priors, type(None)):
        score += np.log(priors)
    idx = np.isfinite(score)
    prob = np.exp(score[idx])
    return entropy(prob / prob.sum())


### Compute the likelihood/correlation score for a tune,
### given the correct key and mode
def score_key_and_mode(tchroma, mode_profiles, target_mode, alg='bayesian', priors=None):
    modes = {m:i for i, m in enumerate(list(MODES.keys()))}

    if '#' in target_mode:
        key_idx = chromatic_map[target_mode[:2]]
        mode_idx = modes[target_mode[2:]]
    else:
        key_idx = chromatic_map[target_mode[0]]
        mode_idx = modes[target_mode[1:]]

    score = profile_correlation(tchroma, mode_profiles, alg=alg)
    return score[key_idx, mode_idx]



def compute_tonal_ambiguity_family(res, parts, mode_profiles, tune_id, p0, meter, factor=4, pid=0.5, nran=10):
    # Establish how many notes should be in a bar, given the factor used
    # to quantize the rhythm, and the meter
    grid_per_bar = int(2 * eval(meter) * factor)

    # Get the MSA for this tune family, and use it to compute the entropy along the sequence
    part_list, msa = PA.get_msa_family(res, parts, tune_id, p0, pid, grid_per_bar, factor=factor)
    ent = PA.get_position_conservation(msa)

    # Sort positions by sequence conservation
    idx = np.argsort(ent)

    # Results container
    out = defaultdict(dict)

    for part, tc in zip(part_list, msa):
        # Compute tonal ambiguity of the whole melody
        out[part]['overall'] = compute_tonal_ambiguity(tc, mode_profiles)

        # Compute tonal ambiguity as a function of the number of notes, for
        #   the first N notes
        #   the N most conserved notes
        #   the N least conserved notes
        #   N random notes
        N_arr = np.arange(10, 110, 10)
        out[part]['first'] = np.zeros(N_arr.size) 
        out[part]['most_cons'] = np.zeros(N_arr.size) 
        out[part]['least_cons'] = np.zeros(N_arr.size) 
        out[part]['random'] = np.zeros(N_arr.size) 
        for i, N in enumerate(N_arr):
            out[part]['first'][i] = compute_tonal_ambiguity(tc[:N], mode_profiles)
            out[part]['most_cons'][i] = compute_tonal_ambiguity(tc[idx][:N], mode_profiles)
            out[part]['least_cons'][i] = compute_tonal_ambiguity(tc[idx][::-1][:N], mode_profiles)
            out[part]['random'][i] = np.mean([compute_tonal_ambiguity(np.random.choice(tc, size=N, replace=True), mode_profiles) for _ in range(nran)])
    return out


def predict_key_family(res, parts, mode_profiles, tune_id, p0, meter, factor=4, pid=0.5, nran=10):
    # Establish how many notes should be in a bar, given the factor used
    # to quantize the rhythm, and the meter
    grid_per_bar = int(2 * eval(meter) * factor)

#   N_arr = np.arange(5, 55, 5)
    N_arr = np.concatenate([np.arange(2, 10, 2), np.arange(10, 55, 5)])

    # Get the MSA for this tune family, and use it to compute the entropy along the sequence
    part_list, msa = PA.get_msa_family(res, parts, tune_id, p0, pid, grid_per_bar, factor=factor)
    if len(part_list) == 0:
        return np.zeros((4, N_arr.size)) * np.nan

    # Get entropy (inverse of sequence convservation)
    ent = PA.get_position_conservation(msa)

    # Sort positions by sequence conservation
    idx = np.argsort(ent)

    # Results container
    out = np.zeros((len(part_list), 4, N_arr.size), bool)

    for i, tc in enumerate(msa):
        for j, N in enumerate(N_arr):
            out[i,0,j] = assign_key_and_mode(tc[:N], mode_profiles)[0] == 'C'
            out[i,1,j] = assign_key_and_mode(tc[idx][:N], mode_profiles)[0] == 'C'
            out[i,2,j] = assign_key_and_mode(tc[idx][::-1][:N], mode_profiles)[0] == 'C'
            out[i,3,j] = np.mean([assign_key_and_mode(np.random.choice(tc, size=N, replace=True), mode_profiles)[0] == 'C' for _ in range(nran)])
    return out.mean(axis=0)


def compare_pearson_and_bayesian(mode_profiles, mode='major', nrep=100):
    N_arr = np.concatenate([np.arange(2, 10, 2), np.arange(10, 55, 5)])
    res = []
    for N in N_arr:
        for _ in range(nrep):
            pitch_vec = np.random.choice(np.arange(12), size=N, replace=True, p=mode_profiles[mode])
            res.append([assign_key_and_mode(pitch_vec, mode_profiles)[0] == 'C',
                        assign_key_and_mode(pitch_vec, mode_profiles, alg='pearson')[0] == 'C'])
    return np.array(res).reshape(N_arr.size, nrep, 2)
        
    






