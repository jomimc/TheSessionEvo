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
from thesession.analysis.substitution import calculate_mutability_and_frequency
from thesession import utils


#######################################################################
### Fig X :: Savage correct substitution matrix

def plot_savage_submat(letters, mat):
    fig, ax = plt.subplots(1,3,figsize=(12,4))

    # Observations
#   plots.plot_submat(letters, mat, ax=ax[0])
#   plots.plot_submat_graph(letters, mat, ax=ax[0], norm=False)

    # Outer product
    diag = np.diagonal(mat)
    offdiag = np.sum(mat, axis=0) - diag
    frequency = offdiag + 2 * diag
    outer = np.outer(frequency, frequency) * 0.1
    np.fill_diagonal(outer, np.diagonal(outer) + 0.9 * diag)
#   plots.plot_submat(letters, outer, ax=ax[1])
#   plots.plot_submat_graph(letters, outer, ax=ax[1], norm=False)

    # Log-odds
#   mat = SM.obs_mat_to_log_odds(mat)
#   plots.plot_submat(letters, mat, ax=ax[2])
#   plots.plot_submat_graph(letters, mat, ax=ax[2], norm=True)


#######################################################################
### Fig X :: Mutability vs Prevalence by mode



def plot_mutability_vs_frequency_modes(mode_alg='exact_pent', alpha=0.5, redo=False, ipid=7, ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(2,4,figsize=(12,6))
        ax = ax.reshape(ax.size)

    for i, mode in enumerate(MODES.keys()):
        path_mat = PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy")
        mat = np.load(path_mat)[ipid]
        mutability, frequency = calculate_mutability_and_frequency(mat)

        ax[i].plot(frequency, mutability, 'o')
        sns.regplot(x=frequency, y=mutability, logx=True, scatter=False, color='k', ax=ax[i])
        ax[i].set_title(mode)

        idx = np.argsort(frequency)[-7:]
        ax[i+4].plot(frequency[idx], mutability[idx], 'o')
        sns.regplot(x=frequency[idx], y=mutability[idx], logx=True, scatter=False, color='k', ax=ax[i+4])
        print(frequency)


    for a in ax:
        a.set_xlabel("Count")
        a.set_ylabel("Mutability")
        a.set_xscale('log')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
#   ax.set_xticks([0,1])
#   ax.set_xticklabels(["All\nNotes", "Within\nScale"])

#   ax.set_yticks(np.arange(5))
#   ax.set_yticklabels(["overall"] + list(MODES.keys()))


#######################################################################
### Fig X :: TheSession substittuion matrix (all tunes)


def plot_submat_graph_overall(ax='', prune=0.01, ipid=7):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(8,4))
    path_mat = PATH_FIG_DATA.joinpath(f"submat-all.npy")
    mat = np.load(path_mat)[ipid]
    letters = np.arange(12)

    main_figs.plot_submat_graph(letters, mat, ax=ax[0], norm=False, prune=prune)
    main_figs.plot_submat_graph(letters, mat, ax=ax[1], norm=True, prune=prune)
    ax[0].set_title("Absolute Rate")
    ax[1].set_title("Normalized Rate")

    
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





