from collections import Counter
import pickle
import shutil 
from subprocess import Popen, PIPE
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from tqdm import tqdm

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.analysis import key_mode as KMF
from thesession.alignment import onset as OA
from thesession.analysis import optimization as OP
from thesession.alignment import parts as PA
from thesession.structure import part_separation as PS
import plots
from thesession.io import savage_loader as savage
from thesession.alignment import pairwise as seq_align
from thesession.io import seq_io
from thesession.analysis import substitution as SM
from thesession import utils


###################################################################################################
### Common functions


### Create folder + files and run mmseqs

### run_mmseqs will not work properly for parts!!!
### To go here or somewhere else? Probably a separate module
def run_mmseqs(df, name, go=4, ge=3, ref='ref', save_fasta=True):
    # First check that mmseqs is installed / in the system's path
    if isinstance(shutil.which('mmseqs'), type(None)):
        raise Exception("MMseqs is not found in the system path. Aborting!")

    # Save to fasta
    path_base = PATH_BASE.joinpath(f'MMseqs/{name}')
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fasta_name = f'all_seq_{name}.fasta'
    if save_fasta:
        path_fasta = path_base.joinpath(fasta_name)
        seq_io.write_all_seq_to_fasta(df.tchroma, df[ref], path_fasta)

    # Save substitution matrix file
    path_submat = PATH_BASE.joinpath(f'MMseqs/{name}/matrix.out')
    submat = SM.basic_submat_A(6, -4)
    SM.write_mmseqs_sub_mat(path_submat, submat, nmax=12)

    # Run mmseqs
    ### Run mmseqs search with inputs:
    ###     fasta (all sequences, for all-vs-all comparison)
    ###     submat (substitution matrix, with match/mismatch scores)
    ###     gap_open, gap_extend (gap penalties, given as positive numbers)
    ### Outputs are saved in path_result, with temporary files in path_tmp
    args = [MMSEQS_BIN, 'easy-search', fasta_name, fasta_name,
            "result.m8", "tmp", '--format-mode', '4',
            '--sub-mat', "matrix.out", '--gap-open', str(go),
            '--gap-extend', str(ge)]

    pipe_output = Popen(args, stdout=PIPE, stderr=PIPE, cwd=str(path_base))
    stdout, stderr = pipe_output.communicate()

    if len(stderr) == 0:
        print("MMseqs has completed successfully!")
        shutil.rmtree(path_base.joinpath("tmp"))
    else:
        print("An error has occurred while running MMseqs!")
        print(stderr)


### Load mmseqs results (or run mmseqs if not done yet)
def load_mmseqs(df, dataset, ref='setting_id', redo=False, annotate=True, save_fasta=True):
    path = PATH_BASE.joinpath(f"MMseqs/{dataset}/result.m8")
    if not path.exists() or redo:
        # Create folder + files and run mmseqs
        run_mmseqs(df, dataset, ref=ref, save_fasta=save_fasta)
    return seq_io.load_mmseqs_pairwise(df, dataset, annotate)



###################################################################################################
### IDENTIFYING SIMILAR TUNES (Fig. 1A)

### Runs on: TheSession, Meertens, Savage et al. (English)
### Loads tunes, converts to standard format:
###     "tchroma" is chroma (12-pitch) representation, transposed to C (int 0)
### tchroma is converted to a 12-letter pitch sequence, and saved to fasta
### Runs mmseqs on tune collections (fasta file)
### Loads mmseqs results and calculates roc curves and auc
### Saves data in the format needed for figures
def data_for_fig1(redo=False):
    # Create container for data for figures
    fig_data = {}

    ### TheSession
    print("Running on TheSession data")

    # Load the full TheSession dataset
    path = PATH_BASE.joinpath("Results/thesession_tunes.pkl")
    if path.exists() and not redo:
        df = pd.read_pickle(path)
    else:
        df, json_data = load_tunes.load_thesession_data_raw()
        df = load_tunes.process_thesession_tunes_pyabc(df, json_data, full=True)
        df.to_pickle(path)

    # Exclude tunes with grace notes, polyphonic pitch, or multiple voices
    # (not considering repeat consistency)
    df = df.loc[~(df.has_grace | df.has_poly | df.has_voices)]

    # First time this will run and load mmseqs results
    # second time onwards will just load mmseqs results, if 'redo' is not set to True
    dataset = "thesession_tunes"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, redo=redo), dataset)
    fig_data = get_total_positives(df, dataset, 'tune_id', fig_data)

    ### Meertens 
    print("Running on Meertens data")
    df = load_tunes.load_meertens_data(redo=redo)[0]

    dataset = "meertens"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, "ref", redo=redo), dataset, fig_data)
    fig_data = get_total_positives(df, dataset, 'song_id', fig_data)

    ### Savage et al. 
    print("Running on data from Savage et al.")
    df = savage.load_savage_df(full=True, redo=redo)
    df = df.loc[df.Language=='English']

    dataset = "savage_english"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, "ref", redo=redo), dataset, fig_data)
    fig_data = get_total_positives(df, dataset, 'chapter', fig_data)

    path = PATH_FIG_DATA.joinpath("fig1_roc_curve_data.pkl")
    pickle.dump(fig_data, open(path, 'wb'))


### Get overall tpr/fpr, accounting for screening stage
def get_total_positives(df, dataset, x='tune_id', fig_data={}):
    total = len(df)**2
    positives = np.sum([n * (n - 1) / 2 for n in df[x].value_counts().values])
    negatives = total - positives
    fig_data[f'{dataset}_positives'] = positives 
    fig_data[f'{dataset}_negatives'] = negatives 
    fig_data[f'{dataset}_total'] = total
    return fig_data 


# Get roc and roc-auc
def get_roc_and_auc(res, dataset, fig_data={}):
    fpr, tpr, _ = roc_curve(res.in_fam, res.fident)
    auc = roc_auc_score(res.in_fam, res.fident)

    # Save to container
    fig_data[f'{dataset}_roc'] = [fpr, tpr]
    fig_data[f'{dataset}_auc'] = auc
    fig_data[f'{dataset}_screened'] = len(res)
    fig_data[f'{dataset}_screened_positives'] = np.sum(res.in_fam)
    fig_data[f'{dataset}_screened_negatives'] = len(res) - np.sum(res.in_fam)
    return fig_data


### Convert the part ID "{tune_id}_{setting_id}_{part_id}"
### to "{tune_id}_{part_id}" for grouping by same tune/part
def get_uniq(s):
    splt = s.split('_')
    return (int(splt[0]), int(splt[-1]))


###################################################################################################
### Note prevalence, mutability and key-finding (Fig. 2)

### Runs on: TheSession
### Loads the full cleaned dataset
### Separates tunes into parts 
### As above, but with parts instead of tunes
###     Parts > tchroma > letters > fasta > run mmseqs
### Loads mmseqs results and analyses similar parts
### Saves data in the format needed for figures
def run_main_alignments(redo=False):
    # Load a cleaned dataset
    # (code takes about 2 hours to run)
    print(f"Loading data (redo is {'on' if redo else 'off'})")
    df, tunes = load_tunes.load_thesession_data(redo=redo)

    # Extract parts
    print(f"Splitting tunes into parts")
    df_parts, parts_data = PS.get_all_parts_thesession(df, tunes, redo=redo)

    # Write parts to fasta
    print(f"Writing to fasta")
    seq_io.write_parts_thesession(parts_data)

    # (Run and ) load mmseqs2 results
    print(f"Running / loading mmseqs")

    ### CHEKC THIS!!! PROBABLY DOES NOT WORK!!!

    res = load_mmseqs(parts_data, "thesession_parts", redo=redo, annotate=False, save_fasta=False)
    print(f"mmseqs gave {len(res)} tune pairs")

    res = PA.prune_identical_parts(res, parts_data)
    print(f"Pruning identical parts leaves {len(res)} tune pairs")

    # Align parts using new algorithm
    res, res0, mismatches = PA.annotate_res(df, res, parts_data, redo=redo)
    print(f"Final set: {len(res0)} tune pairs")

    return df, tunes, df_parts, parts_data, res, res0, mismatches



# Run analyses for Fig 2:
#    Note prevalence, mutability, key finding, IDyOM
def data_for_fig2(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False):

    _ = note_prevalence_mutability(res0, mismatches, tunes, redo=redo)
    _ = note_prevalence_mutability_savage(redo=redo)
    _ = note_stability_key_finding(df, tunes, res0, parts_data, 0.85, redo=redo)


### Get substitution matrices for different PID thresholds
def get_submat_by_pid(res0, mismatches, pid_list, path_mat, alpha=0.5, redo=False):
    if path_mat.exists() and not redo:
        mat = np.load(path_mat)
    else:
        mat = []
        for pid in pid_list:
            idx = np.array(res0.fident > pid, bool)
            obs = PA.subs_to_observations(res0.loc[idx], mismatches.loc[idx], alpha=alpha)
            mat.append(SM.convert_observations_to_matrix(obs, True)[1])
        mat = np.array(mat)
        np.save(path_mat, mat)
    return mat


### Get substitution matrices for different groups of tune parts.
### These can be easily used to calculate prevalence and mutability later.
def note_prevalence_mutability(res0, mismatches, tunes, alpha=0.5, redo=False):

    # Get substitution matrices for:
    #   All
    #   Modes, for each mode compatibility
    #   Dances,
    #   Modes and dances (one mode compatibility)
    #   Each value of PID in np.arange(0.5, 1, 0.05)

    # Get the melodic interval distribution 
    mint = PA.get_mint_dist(tunes)

    # Get substitution matrices
    pid_list = np.arange(0.5, 1, 0.05)
    mat_dict = {}

    # submat: all
    path_mat = PATH_FIG_DATA.joinpath("submat-all.npy")
    mat_dict["all"] = get_submat_by_pid(res0, mismatches, pid_list, path_mat, alpha, redo=redo)
    print(f"{len(res0)} pairs used for All")

    # submat: mode
    mode_alg_list = ['exact', 'loose', 'exact_pent', 'loose_pent']
    for mode_alg in mode_alg_list:
        idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
        for mode, idx in zip(MODES.keys(), idx_list):
            path_mat = PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy")
            k = f"{mode_alg}-{mode}"
            mat_dict[k] = get_submat_by_pid(res0.loc[idx], mismatches.loc[idx], pid_list, path_mat, alpha, redo=redo)
            print(f"{np.sum(idx)} pairs used for {k}")

    # submat: dance
    dance_list = ['reel', 'jig', 'polka', 'hornpipe', 'slip jig', 'slide']
    for dance in dance_list:
        path_mat = PATH_FIG_DATA.joinpath(f"submat-{dance}.npy")
        k = f"{dance}"
        print(f"{np.sum(idx)} pairs used for {k}")
        idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool)
        mat_dict[k] = get_submat_by_pid(res0.loc[idx], mismatches.loc[idx], pid_list, path_mat, alpha, redo=redo)

    # submat: mode and dance
    idx_list = utils.get_mode_indices(res0, mismatches, 'exact_pent')
    for dance in dance_list:
        for mode, idx in zip(MODES.keys(), idx_list):
            path_mat = PATH_FIG_DATA.joinpath(f"submat-{mode}-{dance}.npy")
            k = f"{mode}-{dance}"
            idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool) & idx
            mat_dict[k] = get_submat_by_pid(res0.loc[idx], mismatches.loc[idx], pid_list, path_mat, alpha, redo=redo)
            print(f"{np.sum(idx)} pairs used for {k}")

    return mat_dict


### Get the substitution matrix for the Bronson (British/American) collection
def note_prevalence_mutability_savage(redo=False):
    path = PATH_FIG_DATA.joinpath(f"submat-savage_english.npy")
    if path.exists() and not redo:
        return np.load(path)
    else:
        df = savage.load_savage_df(full=True, redo=False)
        df = df.loc[df.Language=='English']
        obs, letters, mat = savage.get_submat(df.loc[df.Language=='English'])
        np.save(path, mat)
        return mat


### Estimate melody key, using different sets of notes:
### original note order, the most conserved notes, and the least conserved notes
def note_stability_key_finding(df, tunes, res0, parts_data, pid=0.85, redo=False):
    path = PATH_FIG_DATA.joinpath(f"note_stability_key_finding_{pid:4.2f}.npy")
    pid_list = np.arange(0.5, 1, 0.05)
    if path.exists() and not redo:
        return np.load(path)
    else:
        idx = res0.fident >= pid
        res0 = res0.loc[idx]

        meter_key = {t:m for t, m in zip(df.tune_id, df.meter)}
        mode_profiles = KMF.get_modal_profiles(df, tunes)

        # Get lists of exact parts for creating multiple sequence alignments
        # Parts must be the same part number, and the same tune id
        # Only take parts that have 10 or more similar pairs

        count = Counter(get_uniq(x) for x in res0[['query', 'target']].values.ravel())
        candidates = sorted(count.items(), key=lambda x: x[1])[::-1]
        part_set = []
        for (tune_id, part_id), num in candidates:
            if num >= 10:
                part_set.append((tune_id, part_id))

        print(f"Running key finding on {len(part_set)} tunes")

        # Evaluate key finding
        meter_list = [meter_key[t] for (t, p) in part_set]
        correct_key = []
        for (t, p), m in tqdm(zip(part_set, meter_list)):
            correct_key.append(KMF.predict_key_family(res0, parts_data, mode_profiles, t, p, m, factor=4, pid=0.5, nran=10))
        correct_key = np.array(correct_key)
        np.save(path, correct_key)
        return correct_key



###################################################################################################
### Note substitutions (Fig. 3)

    # Run analyses for Fig 3:
    #    Substitution rates + log odds, sub distance (separate by mode, dance, mode and dance, all PID)
def data_for_fig3(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False, mode_alg='exact_pent'):
    _ = note_prevalence_mutability(res0, mismatches, tunes, redo=redo)
    _ = mint_dist(tunes)

    path = PATH_FIG_DATA.joinpath(f"sub_dist_all.npy")
    _ = note_sub_dist(res0, mismatches, parts_data, path, alpha=0.5, redo=redo)
    
    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    for mode, idx in zip(MODES.keys(), idx_list):
        path = PATH_FIG_DATA.joinpath(f"sub_dist_{mode_alg}_{mode}.npy")
        _ = note_sub_dist(res0.loc[idx], mismatches.loc[idx], parts_data, path, alpha=0.5, redo=redo)



### Get the melodic interval distribution
def mint_dist(tunes, redo=False):
    path = PATH_FIG_DATA.joinpath(f"mint_dist_tunes.npy")
    if path.exists() and not redo:
        return np.load(path)
    else:
        mint = PA.get_mint_dist(tunes)
        np.save(path, mint)
        return mint

### Expected substitution distance rate.
### For each tune part, get the expected melodic interval
### distribution one would obtain by shuffling repeatedly.
def get_base_sub_dist_rate(res, mismatches, parts, alpha=0.5):
    M = np.arange(1, 14)
    tot = np.zeros(M.size, float)
    parts_dict = Counter(res[['query', 'target']].values.ravel())
    weights = utils.inverse_frequency_weights(res, alpha)
    for w, (p, c) in zip(weights, parts_dict.items()):
        tmidi = parts[p][0][1].astype(int)
        # Get the difference of all notes with all notes, to get
        # all possible melodic intervals
        count = Counter(np.abs(tmidi[:,None] - tmidi[None,:]).ravel())
        # Should this not also be normalized by melody length???
        for k, v in count.items():
            if k in M:
                tot[k-1] += v * c * w 
    return tot


### Empirical substitution distance rate
def note_sub_dist(res, mismatches, parts, path, alpha=0.5, redo=False):
    if path.exists() and not redo:
        return np.load(path)
    else:
        pid_list = np.arange(0.5, 1, 0.05)
        X = np.arange(1, 14)
        out = []
        for pid in pid_list:
            idx = np.array((res.frac_eq >= pid), bool)

            # Calculate the absolute counts of substitution distances
            # for each tune
            sub_dist = []
            weights = utils.inverse_frequency_weights(res.loc[idx], alpha)
            for sd in mismatches.loc[idx, 'sub_dist']:
                sd_count = Counter(sd)
                sub_dist.append([sd_count.get(i,0) for i in X])

            # Calculate the expected counts of substitution distances
            tot = get_base_sub_dist_rate(res.loc[idx], mismatches.loc[idx], parts, alpha)

            # Sum counts over tunes, with inverse frequency weighting
            Y = np.sum(np.array(sub_dist) * weights[:, None], axis=0)

            # Get the log odds of the ratio of the actual vs expected value
            Y2 = np.log(Y / np.sum(Y)) - np.log(tot / tot.sum())

            out.append([Y, Y2])
        out = np.array(out)
        np.save(path, out)
        return out


### Empirical substitution distance rate for the Bronson (British/American)
### and Japanese collections
def note_sub_dist_savage(redo=False):
    path = PATH_FIG_DATA.joinpath(f"sub_dist_savage.npy")
    if path.exists() and not redo:
        return np.load(path)
    else:
        df = savage.load_savage_df(full=True, redo=redo)
        languages = ['English', 'Japanese']
        out = []
        for i, l in enumerate(languages):
            idx = df.Language == l
            tot, expected = get_sub_dist_savage(df.loc[idx])
            out.append([tot, expected])
        out = np.array(out)
        np.save(path, out)
        return out


def get_sub_dist_savage(df, redo=False):
    M = np.arange(1, 14)
    tot = np.zeros(M.size, float)
    expected = np.zeros(M.size, float)
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

        # Remove indels
        idx1 = np.where(al1 != '-')[0]
        idx2 = np.where(al2 != '-')[0]
        tc1 = tc1[al2[idx1] != '-']
        tc2 = tc2[al1[idx2] != '-']

        # Get substitutions
        sub_idx = np.where(tc1 != tc2)[0]
        sub_dist = np.abs(tc1[sub_idx] - tc2[sub_idx])
        for d, c in Counter(sub_dist).items():
            if d in M:
                tot[d-1] += c

        # Get aligned notes
        for tc in [tc1, tc2]:
            count = Counter(np.abs(tc[:,None] - tc[None,:]).ravel())
            for d, c in count.items():
                if d in M:
                    expected[d-1] += c

    return tot, expected



###################################################################################################
### Sequence position (Fig. 4)

    # Run analyses for Fig 4:
    #    Within-measure / across-measure rates, hierarchy and prevalence, (separate by mode, dance, mode and dance, all PID)
    #    covariance and repetition (separate by mode, dance, mode and dance, all and most common 100 tunes)
def data_for_fig4(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False):
    _ = bar_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False)
    _ = bar_pos_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False)
    _ = onset_histograms(res0, parts_data, redo=redo)
    _ = bar_pos_rate_corr(redo=redo)


### Calculate the average substitution rate in a measure, as a function
### of the position of the measure in the part.
### For many different groups of tune parts.
def bar_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False):
    pid_list = np.arange(0.5, 1, 0.05)
    dance_list = ['reel', 'jig', 'polka', 'hornpipe']
    X = np.arange(8)
    rate_dict = {}

    # Only look at dances that are known to have the 8-bar structure
    idx = np.array(res0.target_dance.isin(dance_list) & res0.query_dance.isin(dance_list), bool)
    res0 = res0.loc[idx]
    mismatches = mismatches.loc[idx]

    # bar rate: all
    path = PATH_FIG_DATA.joinpath("bar_rate-all.npy")
    rate_dict['all'] = get_bar_rate(res0, mismatches, pid_list, path, alpha=alpha, redo=redo)
    print(f"{len(res0)} pairs used for All")

    # bar rate: meter
    # Only run on 4/4, 2/4 and 6/8
    for meter in METER_LIST[:3]:
        idx = np.array((res0.target_meter==meter)&(res0.query_meter==meter), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{meter.replace('/', '_')}.npy")
        rate_dict[meter] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {meter}")

    # bar rate: dance
    for dance in dance_list:
        idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{dance}.npy")
        rate_dict[meter] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {dance}")

    # bar rate: mode
    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    for mode, idx in zip(MODES.keys(), idx_list):
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{mode}.npy")
        rate_dict[mode] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {mode}")

    return rate_dict


### Calculate the average substitution rate in a measure, as a function
### of the position of the measure in the part.
### Do this for different PID thresholds.
def get_bar_rate(res, mismatches, pid_list, path, alpha=0.5, redo=False):
    if path.exists() and not redo:
        return np.load(path)
    else:
        ci = [0.025, 0.975]
        rate_stats = []
        for pid in pid_list:
            weights = utils.inverse_frequency_weights(res, alpha)
            sub_rate_all = get_bar_subrate(mismatches, max_bar=8)
            Y, Ysample = get_bar_subrate_stats(sub_rate_all, weights)
            Ys = np.std(Ysample, axis=0)
            # Save the mean, standard deviation, and the 95% CI
            rate_stats.append([Y, Ys] + list(np.quantile(Ysample, ci, axis=0)))
        rate_stats = np.array(rate_stats)
        np.save(path, rate_stats)
        return rate_stats


### Get bar substitution rate
def get_bar_subrate(mismatches, max_bar=8):
    sub_rate_all = []
    # Calculate substitution rate per bar for each tune
    for sub_bar, grid_per_bar in zip(*mismatches[['sub_bar', 'grid_per_bar']].values.T):
        sub_rate = np.zeros(max_bar)
        for b in sub_bar[sub_bar < max_bar]:
            # The number of substitutions depends on how coarsely the bar
            # has been discretized.
            # Thus, we measure rates in terms of bar fraction,
            # i.e. a rate of 0.5 means notes equal to have the total bar duration are substituted
            sub_rate[b] += 1 / grid_per_bar
        sub_rate_all.append(sub_rate)
    return np.array(sub_rate_all)


### Calculate the substitution rate as a function of the position within a measure.
### For many different groups of tune parts.
def bar_pos_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False):
    pid_list = np.arange(0.5, 1, 0.05)
    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    rate_dict = {}

    # bar pos rate: meter
    for meter in METER_LIST:
        sd = SUBDIV_METER[meter]
        idx = np.array((res0.target_meter==meter)&(res0.query_meter==meter), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{meter.replace('/', '_')}.npy")
        rate_dict[meter] = get_bar_pos_rate(res0.loc[idx], mismatches.loc[idx], sd, pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {meter}")

    # bar pos rate: dance
    for dance in DANCE_LIST:
        sd = SUBDIV_DANCE[dance]
        idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool)
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{dance}.npy")
        rate_dict[meter] = get_bar_pos_rate(res0.loc[idx], mismatches.loc[idx], sd, pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {dance}")

    return rate_dict


### Calculate the substitution rate as a function of the position within a measure.
### For different PID thresholds.
def get_bar_pos_rate(res, mismatches, subdivision, pid_list, path, alpha=0.5, redo=False):
    if path.exists() and not redo:
        return np.load(path)
    else:
        ci = [0.025, 0.975]
        rate_stats = []
        for pid in pid_list:
            idx = np.array(res.fident > pid, bool)
            weights = utils.inverse_frequency_weights(res.loc[idx], alpha)
            # Calculate substitution rate per bar for each tune
            sub_rate_all = []
            for sub_pos, nbar in zip(mismatches.loc[idx, 'sub_pos'], res.loc[idx, 'nbars']):
                # sub_pos is given in units of fraction of total bar duration
                # This converts it to units of (usually eighth notes, but can be finer grained)
                # Rounding takes care of floating point errors (e.g. 1.99999)
                sub_pos_count = Counter(np.round(sub_pos * subdivision, 1))

                # Only include integers, and divide by the number of bars
                # to get rate in units of substitutions per bar
                sub_rate = np.array([sub_pos_count[i] / nbar for i in range(subdivision)])
                sub_rate_all.append(sub_rate)

            Y, Ysample = get_bar_subrate_stats(np.array(sub_rate_all), weights)
            Ys = np.std(Ysample, axis=0)
            # Save the mean, standard deviation, and the 95% CI
            rate_stats.append([Y, Ys] + list(np.quantile(Ysample, ci, axis=0)))
        np.save(path, rate_stats)
        return np.array(rate_stats)


### Bootstrap substitution rates to get confidence intervals
def get_bar_subrate_stats(sub_rate_all, weights, nrep=1000):
    Ysample = []

    # Calculate the weighted mean
    w = weights / weights.sum()
    Y = np.sum(sub_rate_all * w[:,None], axis=0)

    # Calculate errors from bootstrapping
    idx = np.arange(w.size)
    for _ in range(nrep):
        sample = np.random.choice(idx, size=w.size, replace=True)
        Ysample.append(np.sum(sub_rate_all[sample] * w[sample,None] / np.sum(w[sample]), axis=0))
    Ysample = np.array(Ysample)
    return Y, Ysample


### Get the probability of an onset occurring at a point within a measure
def onset_histograms(res0, parts_data, redo=False):
    path = PATH_FIG_DATA.joinpath("onset_histograms.pkl")
    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        ci = [0.025, 0.975]
        hist_stats = {} 
        for meter in METER_LIST:
            idx = (res0.target_meter==meter)&(res0.query_meter==meter)
            hist = get_onset_histograms(res0.loc[idx], parts_data, SUBDIV_METER[meter])
            hist = hist / hist.sum(axis=1)[:,None]
            Ym = np.mean(hist, axis=0)
            Ys = np.std(hist, axis=0)
            Ylo, Yhi = np.quantile(hist, ci, axis=0)
            hist_stats[meter] = [Ym, Ys, Ylo, Yhi]
        pickle.dump(hist_stats, open(path, 'wb'))
        return hist_stats


### Multiply duration values by a factor of 2 to get units of eighth notes
def get_onset_histograms(res, parts_data, subdivision, factor=2):
    onset_count_all = []
    for q, t in zip(*res[['query', 'target']].values.T):
        onset_count = Counter(round(float(x)*factor, 1) % subdivision for y in [q, t] for x in np.cumsum(parts_data[y][0][0]))
        # No need to normalize by bar, since each position can be found in 
        # any bar
        onset_count_all.append(np.array([onset_count.get(i, 0) for i in range(subdivision)]))
    return np.array(onset_count_all)


### Correlation between bar position substitution rate and metrical hierarchy,
### and onset stability 
def bar_pos_rate_corr(redo=False):
    path = PATH_FIG_DATA.joinpath(f"bar_pos_rate_corr.npy")
    if path.exists() and not redo:
        return np.load(path)
    else:
        pid_list = np.arange(0.5, 1, 0.05)
        corr = []
        for ipid in range(pid_list.size):
            df = load_hierarchy_stability_df(ipid)
            corr.append(pearsonr(*df[['hierarchy', 'rel_sub_rate']].values.T)[0]**2)
            corr.append(pearsonr(*df[['rel_stability', 'rel_sub_rate']].values.T)[0]**2)

            X = sm.add_constant(df[['hierarchy', 'end_pos']].values)
            Y = df['rel_sub_rate'].values
            model = sm.OLS(Y, X)
            results = model.fit()
            corr.append(results.rsquared)
        corr = np.array(corr).reshape(pid_list.size, 3)
        np.save(path, corr)
    return corr


### Convert position stability, hierarchy, and onset stability into a dataframe
def load_hierarchy_stability_df(ipid=7):
    path = PATH_FIG_DATA.joinpath(f"onset_histograms.pkl")
    stability = pickle.load(open(path, 'rb'))
    cols = ['hierarchy', 'end_pos', 'meter', 'stability', 'rel_stability',
            'sub_rate', 'rel_sub_rate']
    data = []
    for i, meter in enumerate(METER_LIST):
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{meter.replace('/', '_')}.npy")
        rate = np.load(path)[ipid]
        stab_mean = np.mean(stability[meter][0])
        for j, (r, s) in enumerate(zip(rate[0], stability[meter][0])):
            data.append([HIERARCHY[meter][j], END_POS[meter][j],
                         meter, s, s / stab_mean,
                         r, r / np.mean(rate[0])])
    return pd.DataFrame(data=data, columns=cols)



###################################################################################################
### Sequence covariance (Fig. 5)

    # Run analyses for Fig 5:
    #    covariance and repetition (separate by mode, dance, mode and dance, all and most common 100 tunes)
    #   don't separate by pid. use all available data to get a strong signal
def data_for_fig5(res0, mismatches, redo=False):
    part_covariance(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False)


### Get the covariance matrices for sets of tunes grouped by meter,
### and for a few tune families
def part_covariance(res, parts_data, factor0=2, nbars=8, alpha=0.5, redo=False):
    # Average across tunes with the same meter
    for meter in METER_LIST:
        path = PATH_FIG_DATA.joinpath(f"part_cov-{meter.replace('/', '_')}.npy")
        part_covariance_meter(res, parts_data, meter, path, alpha=alpha, redo=redo)

    # Look at individual parts of tunes,
    # sort by most pairs, and pick the first 10
    res = res.loc[(res.query_tune==res.target_tune)&(res.query_part==res.target_part)]
    count = Counter(get_uniq(x) for x in res[['query', 'target']].values.ravel())
    candidates = sorted(count.items(), key=lambda x: x[1])[::-1]
    part_set = []
    for (tune_id, part_id), num in candidates:
        print(tune_id, part_id, num)
        part_set.append((tune_id, part_id))
        if len(part_set) >= 10:
            break
    for tune_id, part_id in part_set:
        idx = (res.query_tune==tune_id)&(res.query_part==part_id)
        meter = res.loc[idx, 'target_meter'].iloc[0]
        path = PATH_FIG_DATA.joinpath(f"part_cov-{tune_id}_{part_id}.npy")
        part_covariance_meter(res.loc[idx], parts_data, meter, path, alpha=alpha, redo=redo)


### Calculate position-position covariance matrices
### factor0 is set to 2, so that the positions are only checked at
### eighth note positions
def part_covariance_meter(res, parts_data, meter, path, factor0=2, nbars=8, alpha=0.5, redo=False):
    if path.exists() and not redo:
        return np.load(path)
    else:
        grid_per_bar = int(4 * eval(meter) * factor0)
        ngrid = nbars * grid_per_bar

        res = res.loc[(res.target_meter==res.query_meter) & (res.target_meter==meter) 
                       & (res.nbars >= nbars)]
        weights = utils.inverse_frequency_weights(res, alpha)
        print(f"Running covariance analysis on {meter} tunes: {len(res)} total")

        changes = []
        eq = []
        idx = []
        for j, i in enumerate(res.index):
            q, t, f, g = res.loc[i, ['query', 'target', 'factor', 'grid_per_bar']]

            # Convert sequences to a standardized grid
            tc1, tc2, factor = PA.part2grid(parts_data[q][0], parts_data[t][0])
            tc1, tc2 = PA.correct_octave_diff(tc1, tc2)

            # If a larger factor was needed due to the presence of notes smaller than
            # eighth notes, throw away any data on positions off the eighth note positions
            if factor > factor0:
                ratio = int(factor / factor0)
                tc1 = tc1[::ratio]
                tc2 = tc2[::ratio]

            # Check that the correct number of grid points are there
            if (tc1.size < ngrid) or (tc2.size < ngrid):
                continue

            # Remove excess
            tc1, tc2 = tc1[:ngrid], tc2[:ngrid]

            # Find the positions that have the same notes in each sequence,
            # but have different notes across sequences.
            # This indicates where repetition occurs, and where long-range covariance
            # is due to preservation of repetition
            mat = (tc1[:,None] == tc1[None,:]) & (tc2[:,None] == tc2[None,:]) & (tc1[:,None] != tc2[None,:])

            changes.append(mat)
            eq.append(tc1 == tc2)
            idx.append(j)

        return np.array(eq).T, np.array(changes), weights[idx]
        cov = np.cov(np.array(eq).T, aweights=weights[idx])
        rep = np.average(changes, weights=weights[idx], axis=0)
        np.save(path, [cov, rep])

        return cov, rep


### Code for recreating the analyses in the paper.
### Creates and saves data for figures.
def main(redo=False):
    # Run alignments for identifying tunes and analyse results
    data_for_fig1(redo=redo)

    # Run alignments for analysing evolutionary patterns
    df, tunes, df_parts, parts_data, res, res0, mismatches = run_main_alignments(redo=redo)

    # Run analysis for Figure 2
    data_for_fig2(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=redo)

    # Run analysis for Figure 3
    data_for_fig3(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=redo)

    # Run analysis for Figure 4
    data_for_fig4(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=redo)

    # Run analysis for Figure 5
    data_for_fig5(res0, mismatches, redo=redo)


if __name__ == "__main__":

    main(redo=False)



