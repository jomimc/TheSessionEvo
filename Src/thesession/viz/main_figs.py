from collections import Counter, defaultdict
import pickle

from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.patches import Rectangle, Patch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, pearsonr, spearmanr, linregress, false_discovery_control
import seaborn as sns
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

from thesession.alignment import parts as PA
from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.io import savage_loader as savage
from thesession.io import seq_io
from thesession.viz import si_figs
from thesession.analysis import substitution as SM
from thesession.analysis.evaluation import calculate_actual_rates, get_precision_recall, calculate_average_precision
from thesession.analysis.substitution import calculate_mutability_and_frequency
from thesession import utils


# Shared aesthetic constants
SCATTER_COLOR = '#6646A3'   # single-color scatter panels
REG_LINE_COLOR = '#444444'  # dark grey regression lines
METER_PALETTE = dict(zip(METER_LIST, sns.color_palette("Set2", n_colors=len(METER_LIST))))


#######################################################################
### Fig 1 :: tune family recognition


### First run "data_for_fig1" in "main.py"
def plot_roc_curve():
    fig, ax = plt.subplots(figsize=(4,4))
    path = PATH_FIG_DATA.joinpath("fig1_roc_curve_data.pkl")
    data = pickle.load(open(path, 'rb'))
    data_names = ['thesession_tunes', 'meertens', 'savage_english']#, 'savage_japanese']
    lbls = ['TheSession', 'Meertens', 'Savage English']#, 'Savage Japanese']

    for i, name in enumerate(data_names):
        fpr, tpr = data[f"{name}_roc"]
        auc = data[f"{name}_auc"]
        tpr2, fpr2 = calculate_actual_rates(data, name)
#       ax.plot(fpr, tpr, label=f"{lbls[i]}\nAUC={auc:4.2f}")
#       tpr3 = data[f"{name}_screened_positives"] * tpr / data[f"{name}_positives"]
        ax.plot(fpr, tpr, label=f"{lbls[i]}\nAUC={auc:4.2f}")
        print(name)
        print(f"Real TPR: {tpr2}")
        print(f"Real FPR: {fpr2}")
        ave_precision = calculate_average_precision(data, name)
        print(f"Average precision: {ave_precision}")
    ax.plot([0,1],[0,1],'-k')
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("fig1_roc.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig1_roc.pdf"), bbox_inches='tight')


def plot_tune_as_pianoroll(tune, fig='', ax='', meter='', factor=2, max_bar=32):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    grid_per_bar = int(eval(meter) * 4 * factor)
    ngrid = max_bar * grid_per_bar
    tc = utils.get_tchroma_grid(tune['tchroma'], tune['dur'], factor).astype(int)
    tc = tc[:ngrid]

    col = sns.color_palette("husl", 12)
    mat = np.zeros((12, tc.size, 3), float) + 1.0
    for i, j in enumerate(tc):
        mat[j,i] = col[j]

    im = ax.imshow(mat, aspect='auto')
    ax.invert_yaxis()
    ax.set_yticks(range(12))
    ax.set_yticklabels(chromatic_notes)
    if meter != '':
        xticks = np.arange(0, tc.size, grid_per_bar)
        ax.set_xticks(xticks[::4])
        ax.set_xticklabels((np.arange(xticks.size) + 1)[::4])

    ax.set_xlabel("Measure")
    ax.set_ylabel("Pitch")

    plt.tick_params(axis='both', length=0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_example_tune(tunes, tune_id=62, meter='6/8'):
    fig, ax = plt.subplots(figsize=(5.75,2.5))
    plot_tune_as_pianoroll(tunes[tune_id], fig, ax, meter) 
    fig.savefig(PATH_FIG.joinpath("fig1_tune.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig1_tune.pdf"), bbox_inches='tight')




#######################################################################
### Fig 2 :: Mutability


def fig2():
    # A: Amino acid prevalence / mutability
    # B: Note stability and tonal hierarchy (rate vs prevalence, inset: correlation for modes / all)
    # C: Key finding - just the key, not the mode (since that didn't work?) 
    #    split data into high / low stability
    # D: IDyOM - models trained on midi, tchroma, and tchroma with mode

    fig = plt.figure(figsize=(12,8))
    fig.subplots_adjust(wspace=9.0, hspace=0.5)
    gs = GridSpec(5,17, height_ratios=[1,1,.1,1,1])
    ax = [fig.add_subplot(gs[:2,:8]),
          fig.add_subplot(gs[3,:8]), fig.add_subplot(gs[4,:8]),
          fig.add_subplot(gs[:2,9:]), fig.add_subplot(gs[0,14:]),
          fig.add_subplot(gs[3:,9:])]
    
    plot_amino_acid_mutability(ax=ax[0])
    plot_mutability_vs_frequency(alpha=0.5, ipid=7, ax=ax[1:4])
    plot_sub_mat_modes_corr(mode_alg='exact_pent', alpha=0.5,
                            redo=False, ipid=7, ax=ax[4])
    plot_key_finding(ax=ax[5])

    fig.savefig(PATH_FIG.joinpath("fig2.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig2.pdf"), bbox_inches='tight')


def plot_amino_acid_mutability(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    mut_1991 = np.array([100, 83, 104, 86, 44, 84, 77, 50, 91, 103, 54, 72, 93, 51, 58, 117, 107, 25, 50, 98]) / 100
    mut_1978 = np.array([100, 65, 134, 106, 20, 93, 102, 49, 66, 96, 40, 56, 94, 41, 56, 120, 97, 18, 41, 74]) / 100
    freq_1991 = np.array([77, 51, 43, 52, 20, 41, 62, 74, 23, 53, 91, 59, 24, 40, 51, 69, 59, 14, 32, 66]) / 1000
    freq_1978 = np.array([87, 41, 40, 47, 33, 38, 50, 89, 34, 37, 85, 81, 15, 40, 51, 70, 58, 10, 30, 65]) / 1000

    sns.regplot(x=freq_1991, y=mut_1991, ax=ax,
                scatter_kws={'color': SCATTER_COLOR},
                line_kws={'color': REG_LINE_COLOR})
    r, p = pearsonr(freq_1991, mut_1991)
    ax.text(0.10, 0.80, f"$r$ = {r:5.2f}\n$p$ = {p:5.2g}",
            transform=ax.transAxes, fontsize=8)
#   sns.regplot(x=freq_1978, y=mut_1978, label='1978')

    idx = [i for i, a in enumerate(amino_acids) if a in 'CLW']
    dxy = [np.array(x) for x in [[0.005, 0.0], [-0.005, -0.1], [0.005, 0.0]]]

    for i, j in enumerate(idx):
        xy = (freq_1991[j], mut_1991[j])
        ax.annotate(amino_acids[j], xy, xy + dxy[i], arrowprops={'width':0.3, 'headwidth':0.5})

    ax.set_xlabel("Prevalence")
    ax.set_ylabel("Mutability")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title("A  Amino Acid Mutability", loc='center', fontweight='bold')


### Plot mutability vs frequency overall
### ipid = 7 corresponds to 0.85
def plot_mutability_vs_frequency(ipid=7, alpha=0.5, ax='', name="submat-all"):
    path_mat = PATH_FIG_DATA.joinpath(f"{name}.npy")
    mat = np.load(path_mat)[ipid]
    mutability, frequency = calculate_mutability_and_frequency(mat)

    if isinstance(ax, str):
        fig = plt.figure(figsize=(12,4))
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

        gs = GridSpec(2,2)
        ax = [fig.add_subplot(gs[0,0]),
              fig.add_subplot(gs[1,0]),
              fig.add_subplot(gs[:,1])]

    X = np.arange(12)
    lbls = [f'\n{s}' if i % 2 else s for i, s in enumerate(chromatic_notes)]
    col = sns.color_palette('Paired')[:2][::-1] * 6

    ax[0].bar(X, frequency, 0.8, alpha=0.7, ec='k', color=col)
    ax[1].bar(X, mutability, 0.8, alpha=0.7, ec='k', color=col)
    for i in [0,1]:
        ax[i].set_xticks(X)
        ax[i].set_xticklabels(lbls)

    ax[0].set_ylabel("Frequency")
    ax[1].set_ylabel("Mutability")
    ax[0].set_yscale('log')

    ax[2].plot(frequency, mutability, 'o')
    sns.regplot(x=frequency, y=mutability, logx=True, scatter=False, color=sns.color_palette()[5], ax=ax[2])
    ax[2].set_xlabel("Count")
    ax[2].set_ylabel("Mutability")
    ax[2].set_xscale('log')

    idx = np.isfinite(frequency) & np.isfinite(mutability) & (frequency > 0)
    r, p = pearsonr(np.log(frequency[idx]), mutability[idx])
    ax[2].text(0.10, 0.27, f"$r$ = {r:5.2f}\n$p$ = {p:5.2g}", transform=ax[2].transAxes, fontsize=12)

    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)



### Plot correlations between mutability and frequency, separated by mode
def plot_sub_mat_modes_corr(mode_alg='exact_pent', alpha=0.5, redo=False, ipid=7, ax=''):
    pid_list = np.arange(0.5, 1, 0.05)
    path = PATH_FIG_DATA.joinpath(f"mode_stability_hierarchy_corr_{mode_alg}_a{alpha:3.1f}.npy")

    if path.exists() and not redo:
        corr = np.load(path)
    else:
        corr = []
        path_list = [PATH_FIG_DATA.joinpath("submat-all.npy")] + \
                    [PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy") for mode in MODES.keys()]
        for i, path_mat in enumerate(path_list):
            mat_list = np.load(path_mat)
            for j, pid in enumerate(pid_list):
                # Calculate overall correlation
                mutability, frequency = calculate_mutability_and_frequency(mat_list[j])

                # Remove zeros or nans (rare notes)
                idx = np.isfinite(frequency) & np.isfinite(mutability) & (frequency > 0)
                r, p = pearsonr(np.log(frequency[idx]), mutability[idx])
                corr.append([r, p])
                # Only look at scale degrees
                if i > 0:
                    idx = np.argsort(frequency)[-7:]
                    r, p = pearsonr(np.log(frequency[idx]), mutability[idx])
                    corr.append([r, p])
                # Add nans as a placeholder instead of within-scale correlation
                # when not separating pairs by mode
                else:
                    corr.append([np.nan, np.nan])

        corr = np.array(corr).reshape(len(MODES) + 1, pid_list.size, 2, 2)
        np.save(path, corr)
        print(corr.shape)

    if isinstance(ax, str):
        fig, ax = plt.subplots()

    cmap = sns.light_palette('seagreen', as_cmap=True)
    im = ax.imshow(np.abs(corr[:,ipid,:,0]), cmap=cmap, aspect='auto')
    for (i, j), z in np.ndenumerate((corr[:,ipid,:,0])):
        zstr = '' if np.isnan(z) else f'{abs(z):4.2f}'
        ax.text(j, i, zstr, ha='center', va='center')

    ax.set_xticks([0,1])
    ax.set_xticklabels(["All\nNotes", "Within\nScale"])

    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(["overall"] + list(MODES.keys()))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    c = corr[:,ipid,:,1]
    print("p-values after correction")
    print(np.round(false_discovery_control(c[~np.isnan(c)]), 3))

    
def plot_key_finding(ax='', pid=0.85):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    path = PATH_FIG_DATA.joinpath(f"note_stability_key_finding_{pid:4.2f}.npy")
    data = np.load(path)
    N_arr = np.concatenate([np.arange(2, 10, 2), np.arange(10, 55, 5)])
    lbls = ['First Notes', 'Most Conserved Notes', 'Least Conserved Notes']
    col = ['k'] + sns.color_palette()[:2]
    Y = []
    for i, c in enumerate(col):
        ym = np.nanmean(data[:,i], axis=0)
        ys = np.nanstd(data[:,i], axis=0)
        yse = ys / len(data)**0.5
        ax.plot(N_arr, ym, '-', c=c, label=lbls[i])
        ax.errorbar(N_arr, ym, yerr=yse, color=c, fmt='')

        Y.append(ym)

    print(np.round(Y[1] / Y[2], 2))

    xlim = ax.get_xlim()
    ax.plot(xlim, [1/12]*2, ':k', label='random')
    ax.set_xlim(xlim)

    ax.legend(bbox_to_anchor=(0.4, 0.5), frameon=False, fontsize=8)
    ax.set_xlabel("Number of eighth notes used to estimate key")
    ax.set_ylabel("Proportion correct")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



#######################################################################
### Fig 3 :: Note Substitutions


def fig3(ipid=7, mode_alg='exact_pent', alpha=0.5, in_key=False, prune=0.01):
    # A: blosum matrix for amino acids + other protein things (LATERe
    # B: substitution matrix for notes in heatmap form
    # C: substitution matrix for notes in graph form
    # G: substitution rate vs distance

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    gs = GridSpec(3,20)
    ax = [fig.add_subplot(gs[i,j:j+6]) for j in [0,8,14] for i in range(2)] + \
         [fig.add_subplot(gs[2,j:j+9]) for j in [0,11]]

    path_mat = PATH_FIG_DATA.joinpath(f"submat-savage_english.npy")
    mat = np.load(path_mat)
    letters = np.arange(12)

    plot_submat_graph(letters, mat, ax=ax[0], norm=False, prune=prune)
    plot_submat_graph(letters, mat, ax=ax[1], norm=True, prune=prune)
    ax[0].set_title("Absolute Rate")
    ax[1].set_title("Normalized Rate")
    
    path_list = [PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy") for mode in MODES.keys()]

    mat_list = [np.load(path_mat)[ipid] for path_mat in path_list]

    for i, ((mode, scale), mat) in enumerate(zip(MODES.items(), mat_list)):
        letters = np.arange(12)
        if in_key:
            letters = np.arange(7) + 1
            mat = mat[MODES[mode]][:, MODES[mode]]
        plot_submat_graph(letters, mat, norm=True, ax=ax[i+2], prune=prune, mode=scale)
        ax[i+2].set_title(mode)

    plot_sub_dist_both(ax=ax[6:])

    fs = 16
    for i, a in zip([0, 2, 6, 7], 'ABCD'):
        ax[i].text(-0.10, 1.00, a, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("fig3.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig3.pdf"), bbox_inches='tight')


def plot_submat_graph(letters, mat, norm=True, ax='', prune=0.001, edge='rel', color=False, mode=None):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    # Get a copy to prevent overwriting original
    mat = mat.copy()

    # Reorder matrix to get label positions correct
    order = np.roll(np.arange(12), -4)[::-1]
    mat = mat[order][:,order]

    # Set uncommon notes to zero to avoid outliers due to poor statistics
    if prune > 0:
        base_count = np.diag(mat)
        base_prob = base_count / np.nansum(base_count)
        idx = np.where(base_prob < prune)

    if norm:
        # Get log-odds
        mat = SM.obs_mat_to_log_odds(mat)

    if prune > 0:
        for i in idx:
            mat[i] = np.nan
            mat[:,i] = np.nan

    # Remove self edges
    np.fill_diagonal(mat, 0)

    # Create graph
    G = nx.from_numpy_array(mat, create_using=nx.DiGraph)

    # Get weights for edges
    weights = np.array([x for x in mat.T.ravel() if x != 0])

    # Assign new weight sizes based on an ordinal scale
    nweight = 14
    weight_cat = np.logspace(-0.5, 1, nweight+1)
    mask = np.isnan(weights)
    wmin, wmax = np.nanmin(weights), np.nanmax(weights)
    weight_class = np.digitize(weights, np.linspace(wmin-0.01, wmax+0.01, nweight))
    edge_weights = weight_cat[weight_class]
    # Set nan weights to zero weight
    edge_weights[mask] = 0

    if color:
        i, j = np.where(mat.T != 0)
        sub_dist = np.abs(i - j).astype(int)
        sub_dist = np.min([sub_dist, np.abs(12 - sub_dist)], axis=0)
        col = np.array(sns.color_palette())[sub_dist]
    else:
        col = 'grey'
        col = [0.6]*3

    if len(letters) == 12:
        lbls = {i:l for i, l in zip(order, chromatic_notes)}
    else:
        lbls = {i:l for i, l in zip(order, letters)}

    if not isinstance(mode, type(None)):
        nodelist = order[mode]
        lbls = {i:chromatic_notes[j] for i, j in zip(nodelist, mode)}
    else:
        nodelist = order.copy()

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', ax=ax,
                           edgecolors=[0.1]*3, nodelist=nodelist)
    nx.draw_networkx_edges(G, pos, edge_color=col, arrows=True,
                           width=edge_weights, arrowstyle='-', min_source_margin=10,
                           min_target_margin=10, ax=ax)

    nx.draw_networkx_labels(G, pos, lbls, ax=ax)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


### Plot substitution rate vs substitution distance
def plot_sub_dist_sub_rate(data=None, ipid=7, ax='', alpha=0.5):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(12,5))
    X = np.arange(1, 14)

    path = PATH_FIG_DATA.joinpath(f"mint_dist_tunes.npy")
    mint = np.load(path)

    ax[0].plot(mint[0][:X.size], (mint[1] / mint[1].sum())[:X.size], '-k', label='Interval distribution', alpha=0.9, fillstyle='none')

    path = PATH_FIG_DATA.joinpath(f"sub_dist_all.npy")
    Y, Y2 = np.load(path)[ipid]

    ax[0].plot(X, Y / np.sum(Y), '-o', label='TheSession', alpha=0.9, fillstyle='none')
    ax[1].plot(X, Y2, '-o', label='TheSession')

    print("Correlation with M-int distribution:")
    print(pearsonr(Y, mint[1][:X.size]))

    print("Correlation with M-int:")
    print(pearsonr(Y, X))

    xlim = ax[1].get_xlim()
    slope, intercept, r, p = linregress(X, Y2)[:4]
    sns.regplot(x=X, y=Y2, ax=ax[1], scatter=False, color='grey', line_kws={"alpha":0.7},
                label=r"$r$ = " + f"{r:4.2f}" + "\n" + r"$p$ = " + f"{p:5.3f}")
    print("Linear fit:")
    print(slope, intercept, r, p)

    ax[1].plot(xlim, [0,0], ':', color='grey')
    ax[1].set_xlim(xlim)



def plot_sub_dist_both(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(12,5))
    plot_sub_dist_sub_rate(ax=ax)

    X = np.arange(1, 12)
    Y1 = np.loadtxt(PATH_DATA.joinpath("SavageFig/English.txt"), float)[:,1]
    Y2 = np.loadtxt(PATH_DATA.joinpath("SavageFig/Japanese.txt"), float)[:,1]
    ax[0].plot(X, Y1 / Y1.sum(), '-s', label='Bronson', alpha=0.9, fillstyle='none')
    ax[0].plot(X, Y2 / Y2.sum(), '-^', label='Japanese', alpha=0.9, fillstyle='none')

    for i in [0,1]:
        ax[i].legend(loc='best', frameon=False, fontsize=8)
        ax[i].set_xlabel("Intervallic distance (semitones)")
    ax[0].set_ylabel("Substitution Rate\n[Probability]")
    ax[1].set_ylabel("Log-odds score")

    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)



#######################################################################
### Fig 4 :: Position dependence (Site)

def fig4(ipid=7, alpha=0.5):
    fig = plt.figure(figsize=(11,6))
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    gs = GridSpec(2,2, width_ratios=[1.5, 1])
    ax = [fig.add_subplot(gs[i,j]) for j in range(2) for i in range(2)]

    plot_bar_pos_sub_rate(ipid, alpha, ax=ax[:2])
    plot_bar_pos_rate_vs_hierarchy(ipid, ax=ax[2])
    plot_bar_pos_rate_vs_stability(ipid, ax=ax[3])

    fs = 16
    for i, a in enumerate('ABCD'):
        ax[i].text(-0.10, 1.00, a, transform=ax[i].transAxes, fontsize=fs)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("fig4.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig4.pdf"), bbox_inches='tight')


def plot_bar_pos_sub_rate(ipid=7, alpha=0.5, redo=False, ax='',
                          meter_list=['4/4', '6/8']):
    if isinstance(ax, str):
        fig, ax = plt.subplots(2,2, figsize=(11,7))
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        ax = ax.reshape(ax.size)

    col = np.array(sns.color_palette("mako", n_colors=4))
    for i, meter in enumerate(meter_list):
        path = PATH_FIG_DATA.joinpath(f"bar_pos_rate-{meter.replace('/', '_')}.npy")
        rate = np.load(path)[ipid]
        X = np.linspace(0, 1, SUBDIV_METER[meter] + 1)[:-1]
        width = 1 / SUBDIV_METER[meter] / 1.5
        ax[i].bar(X, rate[0], width, color=col[HIERARCHY[meter]], alpha=0.7, ec='k')

        # Manually add whiskers for confidence intervals
        for j in range(SUBDIV_METER[meter]):
            ax[i].plot([X[j]]*2, [rate[2,j], rate[3,j]], '-', color='grey')

        ax[i].set_xticks(X)
        ax[i].set_xticklabels(np.arange(X.size) + 1)
        ax[i].set_ylim(0, ax[i].get_ylim()[1])
        ax[i].set_xlabel("Position in Measure (eighth note)")
        ax[i].set_ylabel("Substitution Rate")
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    handles = [Patch(color=c) for c in col[:3]]
    lbls = ["Main beat", "2nd beat", "3rd beat"]
    ax[0].legend(handles, lbls, frameon=False, loc='upper center', ncol=3,
                 fontsize=8, handletextpad=0.5, columnspacing=1.0)


### Plot bar substitution rate vs meter
def plot_bar_sub_rate(ipid=7, alpha=0.5, max_bar=8, redo=False, ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(figsize=(9,7))

    X = np.arange(max_bar) + 1
    col = [sns.color_palette()[i%4] for i in X]

    path = PATH_FIG_DATA.joinpath(f"bar_rate-all.npy")
    bar_rate = np.load(path)[ipid]

    ax.bar(X, bar_rate[0], 0.8, color=col, alpha=0.7, ec='k')
    for j in range(max_bar):
        ax.plot([X[j]]*2, [bar_rate[2][j], bar_rate[3][j]], '-', color='grey')
    ax.set_xlabel("Bar")
    ax.set_ylabel("Substitution Rate")


def plot_bar_pos_rate_vs_hierarchy(ipid=7, ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    col = np.array(sns.color_palette("mako", n_colors=4))
    df = si_figs.load_hierachy_stability_df(ipid)
    sns.boxplot(x='hierarchy', y='rel_sub_rate', data=df, ax=ax, color='white', showfliers=False)
    sns.stripplot(x='hierarchy', y='rel_sub_rate', data=df, ax=ax, hue='meter',
                  palette=METER_PALETTE)

    r, p = pearsonr(*df[['hierarchy', 'rel_sub_rate']].values.T)
    ax.text(0.05, 0.60, f"$r$ = {r:5.2f}\n$p$ = {p:5.2g}", transform=ax.transAxes, fontsize=8)

    ax.set_xlabel("Metrical Hierarchy")
    ax.set_ylabel("Substitution rate")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles = [Line2D([], [], marker='o', color=METER_PALETTE[m], lw=0) for m in METER_LIST]
    ax.legend(handles, METER_LIST, frameon=False, ncol=2, fontsize=8,
              bbox_to_anchor=(0.75, 1.15), loc='upper right',
              handlelength=1.0, columnspacing=0.8, handletextpad=0.4)


def plot_bar_pos_rate_vs_stability(ipid=7, ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    df = si_figs.load_hierachy_stability_df(ipid)
    sns.scatterplot(x='rel_stability', y='rel_sub_rate', data=df, hue='meter', ax=ax,
                    palette=METER_PALETTE)
    ax.plot(*df.loc[df.end_pos==1, ['rel_stability','rel_sub_rate']].values.T, 'ok', fillstyle='none', ms=10)
    sns.regplot(x='rel_stability', y='rel_sub_rate', data=df, scatter=False, color=REG_LINE_COLOR, ax=ax)

    r, p = pearsonr(*df[['rel_stability', 'rel_sub_rate']].values.T)
    ax.text(0.25, 0.10, f"$r$ = {r:5.2f}\n$p$ = {p:6.4f}", transform=ax.transAxes, fontsize=8)

    ax.set_xlabel("Rhythmic Strength")
    ax.set_ylabel("Substitution Rate")

    handles = [Line2D([], [], marker='o', color=METER_PALETTE[m], lw=0) for m in METER_LIST] + \
              [Line2D([], [], marker='o', fillstyle='none', ms=10, color='k', lw=0)]
    lbls = METER_LIST + ["End position"]
    ax.legend(handles, lbls, frameon=False, ncol=2, fontsize=8,
              bbox_to_anchor=(0.75, 1.15), loc='upper right',
              handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


#######################################################################
### Fig 5 :: Position dependence (Covariance)



def fig5(ipid=7, alpha=0.5):
#   tunes = [(222, 0), (71, 0), (208, 0)]

    fig = plt.figure(figsize=(12,6))
    fig.subplots_adjust(wspace=0.65, hspace=0.0)
    gs = GridSpec(2,4)
    ax = [fig.add_subplot(gs[i,j]) for j in range(4) for i in range(2)]

    plot_cov_part(2, 4, fig=fig, ax=ax[:2])
    plot_cov_meter("4/4", fig=fig, ax=ax[2:4])
    plot_cov_meter("6/8", fig=fig, ax=ax[4:6])
    plot_cov_meter("9/8", fig=fig, ax=ax[6:8], nbars=4)
    
    ttls = ['One Part', 'All 4/4', 'All 6/8', 'All 9/8']
    for i in range(4):
        ax[i*2].set_title(ttls[i])

    fig.savefig(PATH_FIG.joinpath("fig5.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig5.pdf"), bbox_inches='tight')


def plot_cov_mat(fig, ax, mat, nbars=8, nanzero=False, cbar_lbl=''):
    if nanzero:
        mat[mat<10**-10] = np.nan
    finite_vals = mat[np.isfinite(mat)]
    vmax = float(np.max(np.abs(finite_vals))) if len(finite_vals) > 0 else 1.0
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(cbar_lbl, rotation=270, labelpad=14.0)

    nstep = mat.shape[0] // nbars
    ax.set_xticks(np.arange(mat.shape[0])[::nstep])
    ax.set_xticklabels(np.arange(nbars) + 1)
    ax.set_yticks(np.arange(mat.shape[0])[::nstep])
    ax.set_yticklabels(np.arange(nbars) + 1)
    ax.set_xlabel("Position (measures)")
    ax.set_ylabel("Position (measures)")


def plot_cov_part(tune_id, part_id, nbars=8, fig='', ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(9,4))
    path = PATH_FIG_DATA.joinpath(f"part_cov-{tune_id}_{part_id}.npy")
    cov, rep = np.load(path)
    lbl = ['Covariance', 'Repetition covariance']
    for i, mat in enumerate([cov, rep]):
        plot_cov_mat(fig, ax[i], mat, cbar_lbl=lbl[i])


def plot_cov_meter(meter, fig='', ax='', nbars=8):
    if isinstance(ax, str):
        fig, ax = plt.subplots(1,2,figsize=(9,4))
    path = PATH_FIG_DATA.joinpath(f"part_cov-{meter.replace('/', '_')}.npy")
    cov, rep = np.load(path)
    lbl = ['Covariance', 'Repetition covariance']
    for i, mat in enumerate([cov, rep]):
        plot_cov_mat(fig, ax[i], mat, cbar_lbl=lbl[i], nbars=nbars)


def plot_melody_structure(cov, d=2, alg='mds'):
    seed = 15204
    model = MDS(n_components=d, max_iter=3000, eps=1e-9, n_init=1, n_jobs=1, random_state=seed)
    if d == 2:
        fig, ax = plt.subplots()
    elif d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    distance = 1 - (cov - np.nanmin(cov))
    coords = model.fit_transform(distance).T

    new_dist = cdist(coords.T, coords.T)
    print(utils.get_corr(distance.ravel(), new_dist.ravel()))

    col = np.array(sns.color_palette('Paired', n_colors=8))
    c = col[np.arange(coords.shape[1])//8]
    bounds = np.arange(9)  # 8 bins → 9 boundaries
    cmap = ListedColormap(col)
    norm = BoundaryNorm(bounds, cmap.N)

    sm = ScalarMappable(norm=norm, cmap=cmap)

    im = ax.scatter(*coords, c=c, s=80, edgecolors='k', cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=bounds[:-1]+0.5, orientation='vertical')
    cbar.set_ticklabels([f"{i}" for i in range(1, 9)])
    cbar.set_label("Measure")
    ax.plot(*coords, '-k', lw=0.3)
    ax.plot(*coords[:,::8], '*k', ms=13)

    ax.grid(False)
    plt.axis('off')
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")

    draw_orientation_triad(ax, origin=(0.9, 0.9, 0.9))

    ax.view_init(elev=13, azim=-53)
    fig.savefig(PATH_FIG.joinpath("fig5_view1.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig5_view1.pdf"), bbox_inches='tight')

    ax.view_init(elev=79, azim=-142)
    fig.savefig(PATH_FIG.joinpath("fig5_view2.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("fig5_view2.pdf"), bbox_inches='tight')


def draw_orientation_triad(ax, size=0.1, origin=(0.1, 0.1, 0.1)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()

    # basis vectors
    axes = np.eye(3)

    colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]

    for v, c, lbl in zip(axes, colors, labels):
        ax.quiver(
            x0, y0, z0,
            v[0], v[1], v[2],
            length=size,
            color=c,
            arrow_length_ratio=0.25,
            linewidth=2
        )
        ax.text(
            x0 + v[0]*size*1.3,
            y0 + v[1]*size*1.3,
            z0 + v[2]*size*1.3,
            lbl,
            color=c
        )


#######################################################################
### New Fig 2 :: Molecular vs Melodic Evolution




PATH_BLAST_MAFFT = PATH_BASE.joinpath("Src", "blast_mafft")

BLOSUM62_AA_PARASAIL = list("ARNDCQEGHILKMFPSTWYV")  # parasail storage order
BLOSUM62_AA = list("CSTAGPDEQNHRKMILVWYF")           # display order


def plot_note_mutability_scatter(ipid=7, alpha=0.5, ax=None, name="submat-all"):
    """Panel B: scatter of note mutability vs frequency (log scale)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    path_mat = PATH_FIG_DATA.joinpath(f"{name}.npy")
    mat = np.load(path_mat)[ipid]
    mutability, frequency = calculate_mutability_and_frequency(mat)

    valid = np.isfinite(frequency) & np.isfinite(mutability) & (frequency > 0)
    sns.regplot(x=frequency[valid], y=mutability[valid], logx=True,
                scatter=True, ax=ax,
                scatter_kws={'color': SCATTER_COLOR},
                line_kws={'color': REG_LINE_COLOR})
    r, p = pearsonr(np.log(frequency[valid]), mutability[valid])
    ax.text(0.30, 0.80, f"$r$ = {r:5.2f}\n$p$ = {p:5.2g}",
            transform=ax.transAxes, fontsize=8)

    # Annotate C (index 0) and G (index 7); offsets in points keep labels clear
    annot_notes = {'C': (0, (5, 15)), 'G': (7, (-20, -8))}
    arrowprops = dict(arrowstyle='->', color='0.3',
                      connectionstyle='arc3,rad=0.15')
    for note, (i, (dx, dy)) in annot_notes.items():
        ax.annotate(note,
                    xy=(frequency[i], mutability[i]),
                    xytext=(dx, dy),
                    textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    arrowprops=arrowprops)

    ax.set_xlabel("Prevalence")
    ax.set_ylabel("Mutability")
    ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("B  Pitch Mutability", loc='center', fontweight='bold')


def plot_blosum_heatmap(ax=None):
    """Panel C: BLOSUM62 heatmap for the 20 standard amino acids."""
    import parasail
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    mat = np.array(parasail.blosum62.matrix)[:20, :20].astype(float)
    # Reorder to desired display order
    order = [BLOSUM62_AA_PARASAIL.index(aa) for aa in BLOSUM62_AA]
    mat = mat[np.ix_(order, order)]
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones_like(mat, dtype=bool))
    mat[mask] = np.nan
    im = ax.imshow(mat, cmap="RdBu_r")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Log-odds score")
    ticks = np.arange(20)
    ax.set_xticks(ticks[:-1])
    ax.set_xticklabels(BLOSUM62_AA[:-1], fontsize=6)
    ax.set_yticks(ticks[1:])
    ax.set_yticklabels(BLOSUM62_AA[1:], fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Dayhoff group outlines drawn in the upper-triangle white space.
    # Each entry is (i_start, i_end) in display-order index.
    dayhoff_groups = [(1, 5), (6, 9), (10, 12), (13, 16), (17, 19)]
    for i0, i1 in dayhoff_groups:
        ax.plot([i0 - 0.5, i1 + 0.5], [i0 - 0.5, i0 - 0.5], 'k-', lw=1.0)  # top edge
        ax.plot([i1 + 0.5, i1 + 0.5], [i0 - 0.5, i1 + 0.5], 'k-', lw=1.0)  # right edge
        ax.plot([i0 - 0.5, i1 + 0.5], [i1 + 0.5, i1 + 0.5], 'k-', lw=1.0)  # bottom edge
        ax.plot([i0 - 0.5, i0 - 0.5], [i0 - 0.5, i1 + 0.5], 'k-', lw=1.0)  # left edge

    ax.set_title("C  Amino Acid Substitution", loc='center', fontweight='bold')


def plot_submat_heatmap(ipid=7, mode_alg="exact_pent", prune=0.001,
                        mode=None, include_out_of_scale=True, ax=None):
    """
    Panel D: heatmap of the log-odds substitution matrix for a single mode.

    Parameters
    ----------
    ipid : int
        PID-threshold index (default 7 → 0.85).
    mode_alg : str
        Mode-detection algorithm label.
    prune : float
        Notes with marginal frequency below this threshold are masked.
    mode : str or None
        Mode name (e.g. 'major').  None uses the pooled 'all' matrix.
    include_out_of_scale : bool
        If False, rows/columns for out-of-scale notes are dropped entirely.
    ax : matplotlib Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    if mode is None:
#       mat = np.load(PATH_FIG_DATA.joinpath("submat-all.npy"))[ipid]
        mat = np.load(PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy"))[ipid]
        mode_notes = None
    else:
        mat = np.load(PATH_FIG_DATA.joinpath(
            f"submat-{mode_alg}-{mode}.npy"))[ipid]
        mode_notes = MODES[mode]

    # Log-odds
    log_odds = SM.obs_mat_to_log_odds(mat.copy())

    # Prune rare notes
    base_count = np.diag(mat)
    base_prob = base_count / np.nansum(base_count)
    rare = base_prob < prune
    log_odds[rare] = np.nan
    log_odds[:, rare] = np.nan
    # Mask diagonal and upper triangle
    upper = np.triu(np.ones(log_odds.shape, dtype=bool))
    log_odds[upper] = np.nan

    labels = np.array(chromatic_notes)

    if not include_out_of_scale and mode_notes is not None:
        keep = mode_notes
        log_odds = log_odds[np.ix_(keep, keep)]
        labels = labels[keep]
    else:
        keep = np.arange(12)

#   vmax = np.nanmax(np.abs(log_odds))
    im = ax.imshow(log_odds, cmap="RdBu_r")#, vmin=-vmax, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Log-odds score")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=8)
    ax.set_yticks(ticks[1:])
    ax.set_yticklabels(labels[1:], fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("D  Pitch Substitution", loc='center', fontweight='bold')


def _plot_conservation_bars(identity, ax, period=8):
    """
    Minimal conservation bar strip: bars only, no spines or ticks,
    diverging colormap fixed at [0, 1], gridlines every *period* positions.
    """
    pos = np.arange(len(identity))
    y = np.asarray(identity, float) - 0.5

    norm = Normalize(vmin=0, vmax=1)
    colormap = get_cmap("RdBu_r")
    colors = colormap(norm(np.asarray(identity, float)))

    ax.bar(pos, y, width=1.0, color=colors, edgecolor="none")
    ax.axhline(0, color="0.5", linewidth=0.5, zorder=1)

    for gx in np.arange(period - 0.5, len(pos), period):
#       ax.axvline(gx, color="0.7", linewidth=0.5, zorder=0)
        ax.plot([gx]*2, [-.5, .5], color="0.7", linewidth=0.5, zorder=0)

    ax.set_xlim(-0.5, len(pos) - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_protein_conservation(uniprot_rsa, ax_bars, ax_scatter):
    """
    Panel E.
    Top (ax_bars): per-residue sequence identity as a stripped bar plot.
    Bottom (ax_scatter): identity vs mean RSA scatter with regression line.
    """
    base = PATH_BLAST_MAFFT / uniprot_rsa
    df_cons = pd.read_csv(base / f"{uniprot_rsa}_conservation.csv")
    df_rsa  = pd.read_csv(base / f"{uniprot_rsa}_rsa_vs_conservation.csv")

    _plot_conservation_bars(df_cons["identity"].values, ax_bars)
    df_rsa['sub_rate'] = 1 - df_rsa['identity']

    ax_scatter.scatter(df_rsa["mean_rsa"], df_rsa["sub_rate"], s=10, color=SCATTER_COLOR)
    sns.regplot(x="mean_rsa", y="sub_rate", data=df_rsa,
                scatter=False, ax=ax_scatter, color=REG_LINE_COLOR)
    r, p = pearsonr(df_rsa["mean_rsa"], -df_rsa["identity"])
    ax_scatter.text(0.80, 0.30, f"$r$ = {r:5.2f}\n$p$ = {p:5.2g}",
                    transform=ax_scatter.transAxes, fontsize=8)
    ax_scatter.set_xlabel("Structural Exposure")
    ax_scatter.set_ylabel("Substitution Rate")
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    ax_bars.set_title("E  Position Conservation", loc='center', fontweight='bold')


def plot_melody_conservation(tune_id, part_id, ax_bars, ax_scatter, ipid=7):
    """
    Panel F.
    Top (ax_bars): per-position melodic conservation as a stripped bar plot.
    Bottom (ax_scatter): metrical stability vs substitution rate scatter.
    """
    path = PATH_FIG_DATA / f"part_cov-{tune_id}_{part_id}_conservation.npy"
    identity = np.load(path)
    _plot_conservation_bars(identity, ax_bars)

    plot_bar_pos_rate_vs_stability(ipid=ipid, ax=ax_scatter)
    ax_bars.set_title("F  Position Conservation", loc='center', fontweight='bold')


def plot_contact_map(uniprot_contact, dist_threshold=8, ax=None,
                     beta_strand=((15, 80), (27, 91))):
    """Panel G: binary Cα contact map (distance < dist_threshold).

    Parameters
    ----------
    beta_strand : pair of (x, y) tuples, optional
        Start and end of a parallel beta-strand contact stripe to highlight
        with a red line (drawn on both sides of the diagonal).
        Set to None to suppress.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    dist = np.load(
        PATH_BLAST_MAFFT / uniprot_contact /
        f"{uniprot_contact}_ca_distances.npy"
    )
    contact = (dist < dist_threshold).astype(float)
    np.fill_diagonal(contact, np.nan)

    ax.imshow(contact, cmap="Greys", interpolation="nearest")

    if beta_strand is not None:
        (x0, y0), (x1, y1) = beta_strand
        kw = dict(color="red", linewidth=2, solid_capstyle="round")
        ax.plot([x0, x1], [y0, y1], **kw)   # upper-left copy
        ax.plot([y0, y1], [x0, x1], **kw)   # lower-right mirror

    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Sequence Position")
    ax.set_title("G  Protein Contact Map", loc='center', fontweight='bold')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_cov_heatmap(meter, ax=None, fig=None):
    """Panel H: covariance heatmap for one tune-part family."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    path = PATH_FIG_DATA / f"part_cov-{meter.replace('/','_')}.npy"
    cov, _ = np.load(path)
    plot_cov_mat(fig or ax.figure, ax, cov, cbar_lbl="Covariance")
    ax.set_title("H  Melody Covariance", loc='center', fontweight='bold')


def new_fig2(uniprot_rsa="P00004", uniprot_contact="P0AA25",
             pos_tune_id=222, pos_tune_part=0,
             dist_threshold=15,
             ipid=7, alpha=0.5,
             meter='4/4',
             mode_alg="exact_pent", mode="major",
             prune=0.001, include_out_of_scale=False):
    # Use subfigures so each cell in the 4x2 grid is fully independent:
    # colorbars, tick labels, and axis labels within one panel cannot
    # shift any other panel.  Panel labels are placed in subfigure
    # coordinates so they always sit at the true top-left of each cell.
    fig = plt.figure(figsize=(8, 9.5), layout='constrained')
    subfigs = fig.subfigures(4, 2, hspace=0.06, wspace=0.06)

    # --- A: amino acid mutability vs frequency ---
    ax_A = subfigs[0, 0].subplots()
    plot_amino_acid_mutability(ax=ax_A)

    # --- B: note mutability vs frequency (scatter only) ---
    ax_B = subfigs[0, 1].subplots()
    plot_note_mutability_scatter(ipid=ipid, alpha=alpha, ax=ax_B)

    # --- C: BLOSUM62 heatmap ---
    ax_C = subfigs[1, 0].subplots()
    plot_blosum_heatmap(ax=ax_C)

    # --- D: melodic substitution matrix heatmap ---
    ax_D = subfigs[1, 1].subplots()
    plot_submat_heatmap(ipid=ipid, mode_alg=mode_alg, prune=prune,
                        mode=mode, include_out_of_scale=include_out_of_scale,
                        ax=ax_D)

    # --- E: protein conservation strip + identity vs RSA scatter ---
    # height_ratios keeps the bar strip thin; constrained_layout handles spacing
    ax_E_bars, ax_E_scatter = subfigs[2, 0].subplots(
        2, 1, height_ratios=[1, 4], gridspec_kw={'hspace': 0.00})
    plot_protein_conservation(uniprot_rsa, ax_E_bars, ax_E_scatter)

    # --- F: melodic conservation strip + stability vs sub-rate scatter ---
    ax_F_bars, ax_F_scatter = subfigs[2, 1].subplots(
        2, 1, height_ratios=[1, 4], gridspec_kw={'hspace': 0.00})
    plot_melody_conservation(pos_tune_id, pos_tune_part,
                             ax_F_bars, ax_F_scatter, ipid=ipid)

    # --- G: Cα contact map ---
    ax_G = subfigs[3, 0].subplots()
    plot_contact_map(uniprot_contact, dist_threshold=dist_threshold, ax=ax_G)

    # --- H: sequence covariance heatmap ---
    # Pass the subfigure so the colorbar is scoped to that cell
    ax_H = subfigs[3, 1].subplots()
    plot_cov_heatmap(meter, ax=ax_H, fig=subfigs[3, 1])

    # Remove per-panel titles (replaced by border labels)
    for sf in subfigs.ravel():
        for ax in sf.get_axes():
            ax.set_title('')

    # Panel letters at the top-left of each subfigure cell
    for sf, lbl in zip(subfigs.ravel(), "ABCDEFGH"):
        bb = sf._subplotspec.get_position(fig)
        fig.text(bb.x0, bb.y1, lbl, fontsize=14, fontweight="bold",
                 va="top", ha="left", clip_on=False)

    # Column header labels centred above each column
    for col_idx, lbl in enumerate(["Proteins", "Melodies"]):
        bb = subfigs[0, col_idx]._subplotspec.get_position(fig)
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.008, lbl,
                 fontsize=15, fontweight='bold',
                 ha='center', va='bottom', clip_on=False)

    # Row labels: vertical text to the left of each row
    row_labels = ["Mutability", "Substitution",
                  "Position\nConservation", "Structure &\nCovariance"]
    for row_idx, lbl in enumerate(row_labels):
        bb = subfigs[row_idx, 0]._subplotspec.get_position(fig)
        fig.text(bb.x0 - 0.035, (bb.y0 + bb.y1) / 2, lbl,
                 fontsize=13, fontweight='bold',
                 ha='center', va='center', rotation=90, clip_on=False)

    fig.savefig(PATH_FIG.joinpath("fig2.png"), bbox_inches="tight")
    fig.savefig(PATH_FIG.joinpath("fig2.pdf"), bbox_inches="tight")
    


def new_fig3(ipid=7, cov_meter="6/8", bar_meter="6/8"):
    fig = plt.figure(figsize=(8, 10), layout='constrained')
    subfigs = fig.subfigures(4, 2, hspace=0.10, wspace=0.06)

    # --- A: key-finding accuracy (left cell only; right cell left empty) ---
    ax_A = subfigs[0, 0].subplots()
    plot_key_finding(ax=ax_A)

    # --- B-C: substitution distance vs rate ---
    ax_B = subfigs[1, 0].subplots()
    ax_C = subfigs[1, 1].subplots()
    plot_sub_dist_both(ax=[ax_B, ax_C])

    # --- D: within-bar substitution rate ---
    ax_D = subfigs[2, 0].subplots()
    plot_bar_pos_sub_rate(ipid=ipid, ax=[ax_D], meter_list=[bar_meter])
    ax_D.set_ylim(0, 0.19)

    # --- E: substitution rate vs metrical hierarchy ---
    ax_E = subfigs[2, 1].subplots()
    plot_bar_pos_rate_vs_hierarchy(ipid=ipid, ax=ax_E)

    # --- F-G: covariance and repetition covariance ---
    # Call plot_cov_mat directly so each colorbar is scoped to its subfigure
    ax_F = subfigs[3, 0].subplots()
    ax_G = subfigs[3, 1].subplots()
    path = PATH_FIG_DATA.joinpath(f"part_cov-{cov_meter.replace('/', '_')}.npy")
    cov, rep = np.load(path)
    plot_cov_mat(subfigs[3, 0], ax_F, cov, cbar_lbl='Covariance')
    plot_cov_mat(subfigs[3, 1], ax_G, rep, cbar_lbl='Repetition covariance')

    # Panel labels on the main figure at the true top-left of each subfigure cell
    used = [subfigs[0, 0], subfigs[1, 0], subfigs[1, 1],
            subfigs[2, 0], subfigs[2, 1], subfigs[3, 0], subfigs[3, 1]]
    for sf, lbl in zip(used, "ABCDEFG"):
        bb = sf._subplotspec.get_position(fig)
        fig.text(bb.x0, bb.y1, lbl, fontsize=14, fontweight="bold",
                 va="top", ha="left", clip_on=False)

    fig.savefig(PATH_FIG.joinpath("fig3.png"), bbox_inches="tight")
    fig.savefig(PATH_FIG.joinpath("fig3.pdf"), bbox_inches="tight")

