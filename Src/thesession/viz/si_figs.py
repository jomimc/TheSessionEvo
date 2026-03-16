from collections import Counter, defaultdict
import pickle

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import parasail
from scipy.stats import gaussian_kde, pearsonr, spearmanr, linregress, false_discovery_control
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

from thesession.alignment import parts as PA
from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.viz import main_figs
from thesession.io import savage_loader as savage
from thesession.io import seq_io
from thesession.analysis import substitution as SM
from thesession.analysis.optimization import load_results_mmseqs
from thesession.analysis.substitution import calculate_mutability_and_frequency
from thesession import utils


#######################################################################
### Fig 1 :: sequence alignment score optimization


def plot_optimization_scores():
    df = load_results_mmseqs()
    fig, ax = plt.subplots(1,2,figsize=(9,4))
    df['mat_type'] = df['mat_type'].map({'A':'equal mismatch', 'B':'linear mismatch'})

    sns.stripplot(x='mat_type', y='auc', data=dfr, hue='mat_type', ax=ax[1])
    ax[0].set_xlabel("Substitution Matrix")
    ax[0].set_ylabel("ROC AUC")

    sns.scatterplot(x='actual_fpr', y='actual_tpr', data=dfr, hue='mat_type', ax=ax[1])
    ax[1].set_xlabel("Actual False Positive Rate")
    ax[1].set_ylabel("Actual True Positive Rate")
    ax[1].legend(loc='best', frameon=False)

    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("si1_roc.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si1_roc.pdf"), bbox_inches='tight')
    


#######################################################################
### SI Fig 2 :: Mutability and prevalence by mode


def plot_mutability_by_mode(mode_alg='exact_pent', ipid=7):
    """
    Four-row panel showing note mutability and prevalence for each mode.

    Each row covers one mode (major, minor, mixolydian, dorian) and
    contains three panels:
      - Top-left:    Count (prevalence) bar chart for all 12 chromatic notes.
      - Bottom-left: Mutability bar chart for all 12 chromatic notes.
      - Right:       Mutability vs Count scatter with two regression lines —
                     one for all notes with non-zero count, and one restricted
                     to the notes that belong to the mode's scale.

    Parameters
    ----------
    mode_alg : str, optional
        Mode-detection algorithm label used in the figure-data filename.
        Default is ``'exact_pent'``.
    ipid : int, optional
        Index into the PID-threshold axis of the saved matrix array.
        Index 7 corresponds to PID = 0.85.  Default is 7.
    """
    mode_list = ['major', 'minor', 'mixolydian', 'dorian']

    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(8, 2, figure=fig, wspace=0.35, hspace=1.0)

    X = np.arange(12)
    lbls = [f'\n{s}' if i % 2 else s for i, s in enumerate(chromatic_notes)]
    bar_col = sns.color_palette('Paired')[:2][::-1] * 6
    col_all  = sns.color_palette()[0]
    col_mode = sns.color_palette()[1]

    for i, mode in enumerate(mode_list):
        path_mat = PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy")
        mat = np.load(path_mat)[ipid]
        mutability, frequency = calculate_mutability_and_frequency(mat)

        mode_notes = MODES[mode]
        idx_all = np.isfinite(frequency) & np.isfinite(mutability) & (frequency > 0)

        # --- Count bar chart (top-left) ---
        ax_count = fig.add_subplot(gs[2 * i, 0])
        ax_count.bar(X, frequency, 0.8, alpha=0.7, ec='k', color=bar_col)
        ax_count.set_xticks(X)
        ax_count.set_xticklabels(lbls)
        ax_count.set_ylabel("Count")
        ax_count.set_yscale('log')
        ax_count.set_title(mode.capitalize())
        ax_count.text(-0.15, 1.05, 'ABCD'[i], transform=ax_count.transAxes,
                      fontsize=14, fontweight='bold', va='top')

        # --- Mutability bar chart (bottom-left) ---
        ax_mut = fig.add_subplot(gs[2 * i + 1, 0])
        ax_mut.bar(X, mutability, 0.8, alpha=0.7, ec='k', color=bar_col)
        ax_mut.set_xticks(X)
        ax_mut.set_xticklabels(lbls)
        ax_mut.set_ylabel("Mutability")

        # --- Mutability vs Count scatter (right, full row height) ---
        ax_scat = fig.add_subplot(gs[2 * i:2 * i + 2, 1])

        # All non-zero notes
        ax_scat.plot(frequency[idx_all], mutability[idx_all], 'o',
                     color=col_all, alpha=0.7, label='All notes')
        sns.regplot(x=frequency[idx_all], y=mutability[idx_all],
                    logx=True, scatter=False, color=col_all, ax=ax_scat)

        # Mode scale notes only
        idx_mode = mode_notes[np.isfinite(mutability[mode_notes]) &
                               (frequency[mode_notes] > 0)]
        ax_scat.plot(frequency[idx_mode], mutability[idx_mode], 'o',
                     color=col_mode, alpha=0.7, label=f'{mode.capitalize()} scale')
        sns.regplot(x=frequency[idx_mode], y=mutability[idx_mode],
                    logx=True, scatter=False, color=col_mode, ax=ax_scat)

        ax_scat.set_xlabel("Count")
        ax_scat.set_ylabel("Mutability")
        ax_scat.set_xscale('log')
        ax_scat.set_title(mode.capitalize())
        ax_scat.legend(loc='best', frameon=False)

        for a in [ax_count, ax_mut, ax_scat]:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("si2_mutability_by_mode.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si2_mutability_by_mode.pdf"), bbox_inches='tight')


#######################################################################
### SI Fig 3 :: Overall substitution graph (TheSession tune parts)


def plot_submat_overall(prune=0.01, ipid=7):
    """
    Unnormalized and normalized substitution graphs for TheSession tune
    parts, pooled across all modes.

    Parameters
    ----------
    prune : float, optional
        Notes with a marginal frequency below this threshold are suppressed.
        Default is 0.01.
    ipid : int, optional
        Index into the PID-threshold axis of the saved matrix array.
        Index 7 corresponds to PID = 0.85.  Default is 7.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3)

    path_mat = PATH_FIG_DATA.joinpath(f"submat-all.npy")
    mat = np.load(path_mat)[ipid]
    letters = np.arange(12)

    main_figs.plot_submat_graph(letters, mat, ax=ax[0], norm=False, prune=prune)
    main_figs.plot_submat_graph(letters, mat, ax=ax[1], norm=True, prune=prune)
    ax[0].set_title("Absolute Rate")
    ax[1].set_title("Normalized Rate")

    for a, lbl in zip(ax, 'AB'):
        a.text(-0.10, 1.00, lbl, transform=a.transAxes, fontsize=16, fontweight='bold')

    fig.savefig(PATH_FIG.joinpath("si3_submat_overall.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si3_submat_overall.pdf"), bbox_inches='tight')



    
#######################################################################
### Fig X :: Substitution matrix correlations


def plot_submat_corr():
    fig, ax = plt.subplots(1,4,figsize=(16,4))
    path_mat = PATH_FIG_DATA.joinpath(f"submat-all.npy")
    mat = np.load(path_mat)
    obs = [SM.obs_mat_to_log_odds(m) for m in mat]
    N = len(obs)
    corr = np.array([[utils.get_corr(obs[i].ravel(), obs[j].ravel()) for i in range(N)] for j in range(N)]) 
    im = ax[0].imshow(corr)
    fig.colorbar(im, ax=ax[0])
    sns.kdeplot(corr.ravel(), label='PID', ax=ax[3])

    ipid = 7
    mode_alg = 'exact_pent'
    path_list = [PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy") for mode in MODES.keys()]
    obs = [SM.obs_mat_to_log_odds(np.load(path)[ipid]) for path in path_list]
    N = len(obs)
    corr = np.array([[utils.get_corr(obs[i].ravel(), obs[j].ravel()) for i in range(N)] for j in range(N)]) 
    im = ax[1].imshow(corr)
    fig.colorbar(im, ax=ax[1])
    sns.kdeplot(corr.ravel(), label='Mode', ax=ax[3])

    blosum_mat = [getattr(parasail, f"blosum{x}").matrix for x in np.arange(35, 95, 5)]
    N = len(blosum_mat)
    corr = np.array([[utils.get_corr(blosum_mat[i].ravel(), blosum_mat[j].ravel()) for i in range(N)] for j in range(N)]) 
    im = ax[2].imshow(corr)
    fig.colorbar(im, ax=ax[2])
    sns.kdeplot(corr.ravel(), label='BLOSUM', ax=ax[3])

    ipid = 7
    mode_alg = 'exact_pent'
    dance_list = ['reel', 'jig', 'polka', 'hornpipe', 'slip jig', 'slide']
    path_list = [PATH_FIG_DATA.joinpath(f"submat-{dance}.npy") for dance in dance_list]
    obs = [SM.obs_mat_to_log_odds(np.load(path)[ipid]) for path in path_list]
    N = len(obs)
    corr = np.array([[utils.get_corr(obs[i].ravel(), obs[j].ravel()) for i in range(N)] for j in range(N)]) 
    sns.kdeplot(corr.ravel(), label='Dance', ax=ax[3])



#######################################################################
### Fig X :: Sub rate vs Sub dist by mode

def plot_sub_dist_mode(ax='', mode_alg='exact_pent', ipid=7):
    if isinstance(ax, str):
        fig, ax = plt.subplots(figsize=(6,5))

    X = np.arange(1, 14)
    
    for i, mode in enumerate(MODES.keys()):
        path = PATH_FIG_DATA.joinpath(f"sub_dist_{mode_alg}_{mode}.npy")
        Y, Y2 = np.load(path)[ipid]

        ax.plot(X, Y2, '-o', label=mode)

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Substitution distance (semitones)")
    ax.set_ylabel("Normalized Rate")


#######################################################################
### Fig X :: Sub dist - M-int corr by PID


def plot_sub_dist_mint_corr(ax='', mode_alg='exact_pent'):
    if isinstance(ax, str):
        fig, ax = plt.subplots(figsize=(6,5))
    path = PATH_FIG_DATA.joinpath(f"mint_dist_tunes.npy")
    mint = np.load(path)[1][:13]

    pid_list = np.arange(0.5, 1, 0.05)

    path = PATH_FIG_DATA.joinpath(f"sub_dist_all.npy")
    Y_list = np.load(path)[:,1]
    corr = np.array([utils.get_corr(Y, mint) for Y in Y_list])
    ax.plot(pid_list, corr, label='All')

    for i, mode in enumerate(MODES.keys()):
        path = PATH_FIG_DATA.joinpath(f"sub_dist_{mode_alg}_{mode}.npy")
        Y_list = np.load(path)[:,1]
        corr = np.array([utils.get_corr(Y, mint) for Y in Y_list])
        ax.plot(pid_list, corr, label=mode)

    ax.set_xlabel("PID")
    ax.set_ylabel("Corr(Sub rate, M-Int dist)")
    ax.legend(loc='best', frameon=False)



#######################################################################
### Fig X :: bar rate vs dance type

def plot_bar_rate_dance(ipid=7, max_bar=8):
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    X = np.arange(max_bar) + 1
    col = [sns.color_palette()[(i-1)%4] for i in X]
    width = 0.8
    for i, dance in enumerate(DANCE_LIST):
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{dance}.npy")
        bar_rate = np.load(path)[ipid]
        ax[i].bar(X, bar_rate[0], width, color=col, alpha=0.7, ec='k')
        # Manually add whiskers for confidence intervals
        for j in range(max_bar):
            ax[i].plot([X[j]]*2, [bar_rate[2,j], bar_rate[3,j]], '-', color='grey')
        ax[i].set_title(dance)

        
def plot_bar_rate_meter(ipid=7, max_bar=8):
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    meter_list = ['4/4', '2/4', '6/8']
    X = np.arange(max_bar) + 1
    col = [sns.color_palette()[(i-1)%4] for i in X]
    width = 0.8
    for i, meter in enumerate(meter_list):
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{meter.replace('/', '_')}.npy")
        bar_rate = np.load(path)[ipid]
        print(bar_rate.shape)
        ax[i].bar(X, bar_rate[0], width, color=col, alpha=0.7, ec='k')
        # Manually add whiskers for confidence intervals
        for j in range(max_bar):
            ax[i].plot([X[j]]*2, [bar_rate[2,j], bar_rate[3,j]], '-', color='grey')
        ax[i].set_title(meter)


def plot_bar_rate_mode(ipid=7, max_bar=8):
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    mode_list = list(MODES.keys())
    X = np.arange(max_bar) + 1
    col = [sns.color_palette()[(i-1)%4] for i in X]
    width = 0.8
    for i, mode in enumerate(mode_list):
        path = PATH_FIG_DATA.joinpath(f"bar_rate-{mode}.npy")
        bar_rate = np.load(path)[ipid]
        print(bar_rate.shape)
        ax[i].bar(X, bar_rate[0], width, color=col, alpha=0.7, ec='k')
        # Manually add whiskers for confidence intervals
        for j in range(max_bar):
            ax[i].plot([X[j]]*2, [bar_rate[2,j], bar_rate[3,j]], '-', color='grey')
        ax[i].set_title(mode)

        
#######################################################################
### Fig X :: bar pos rate vs dance type

        
def plot_bar_pos_rate_dance(ipid=7):
    fig, ax = plt.subplots(2,2)
    ax = ax.reshape(ax.size)
    col = np.array(sns.color_palette("mako", n_colors=4))
    for i, dance in enumerate(DANCE_LIST):
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{dance}.npy")
        rate = np.load(path)[ipid]
        X = np.linspace(0, 1, SUBDIV_DANCE[dance] + 1)[:-1]
        width = 1 / SUBDIV_DANCE[dance] / 1.5
        ax[i].bar(X, rate[0], width, color=col[HIERARCHY_DANCE[dance]], alpha=0.7, ec='k')
        # Manually add whiskers for confidence intervals
        for j in range(SUBDIV_DANCE[dance]):
            ax[i].plot([X[j]]*2, [rate[2,j], rate[3,j]], '-', color='grey')
        ax[i].set_title(dance)


def plot_bar_pos_rate_meter(ipid=7):
    fig, ax = plt.subplots(2,3)
    ax = ax.reshape(ax.size)

    col = np.array(sns.color_palette("mako", n_colors=4))
    for i, meter in enumerate(METER_LIST):
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{meter.replace('/', '_')}.npy")
        rate = np.load(path)[ipid]
        X = np.linspace(0, 1, SUBDIV_METER[meter] + 1)[:-1]
        width = 1 / SUBDIV_METER[meter] / 1.5
        ax[i].bar(X, rate[0], width, color=col[HIERARCHY[meter]], alpha=0.7, ec='k')
        # Manually add whiskers for confidence intervals
        for j in range(SUBDIV_METER[meter]):
            ax[i].plot([X[j]]*2, [rate[2,j], rate[3,j]], '-', color='grey')
        ax[i].set_title(meter)


def load_hierachy_stability_df(ipid=7):
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


def plot_bar_pos_rate_vs_hierarchy(ipid=7):
    fig, ax = plt.subplots()
    col = np.array(sns.color_palette("mako", n_colors=4))
    df = load_hierachy_stability_df(ipid)
    sns.boxplot(x='hierarchy', y='rel_sub_rate', data=df, ax=ax, color='white', showfliers=False)
    sns.stripplot(x='hierarchy', y='rel_sub_rate', data=df, ax=ax, hue='meter')
    ax.set_xlabel("Metrical hierarchy")
    ax.set_ylabel("Relative substitution rate")


def plot_bar_pos_rate_vs_stability(ipid=7):
    fig, ax = plt.subplots()
    df = load_hierachy_stability_df(ipid)
#   for i, meter in enumerate(METER_LIST):
    sns.scatterplot(x='rel_stability', y='rel_sub_rate', data=df, hue='meter')
    sns.regplot(x='rel_stability', y='rel_sub_rate', data=df, scatter=False, color='k')
    r, p = pearsonr(*df[['rel_stability', 'rel_sub_rate']].values.T)
    print(r, p)
    ax.set_xlabel("Metrical stability (onset probability)")
    ax.set_ylabel("Relative substitution rate")
    

def plot_bar_pos_rate_corr():
    fig, ax = plt.subplots()
    pid_list = np.arange(0.5, 1, 0.05)
    path = PATH_FIG_DATA.joinpath(f"bar_pos_rate_corr.npy")
    corr = np.load(path)
    ax.plot(pid_list, corr[:,0], label='Hierarchy')
    ax.plot(pid_list, corr[:,1], label='Hierarchy + EndPos')
    ax.plot(pid_list, corr[:,2], label='Stability')
        
        

def mutability_vs_prevalence_savage(df):
    languages = ['English', 'Japanese']
    fig, ax = plt.subplots(1,2,figsize=(9,4))
    for i, lang in enumerate(languages):
        obs, letters, mat = savage.get_submat(df.loc[df.Language==lang])
        mutability, frequency = calculate_mutability_and_frequency(mat)
        idx = frequency > 100
        sns.regplot(x=frequency[idx], y=mutability[idx], ax=ax[i])
        ax[i].set_title(lang)
        print(lang)
        print(np.sum(idx))
        print(utils.get_corr(np.log(frequency[idx]), mutability[idx], p=1))





