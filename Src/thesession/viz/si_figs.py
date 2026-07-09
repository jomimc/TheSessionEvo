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
from thesession.config import (
    DANCE_LIST, HIERARCHY, HIERARCHY_DANCE, METER_LIST, MODES, PATH_FIG,
    PATH_FIG_DATA, SUBDIV_DANCE, SUBDIV_METER, chromatic_notes, letters,
)
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
    fig, ax = plt.subplots(3,3,figsize=(12,12))
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    datasets = ['thesession_tunes', 'meertens', 'savage_english']
    ttls = ['TheSession', 'Meertens', 'British/American']
    xlbls = ['equal\nmismatch', 'linear\nmismatch', 'chosen\nscore']
    for i, dataset in enumerate(datasets):
        df = load_results_mmseqs(dataset)
        idx = (df.gap_open==4)&(df.gap_extend==3)&(df.mat_type=='A')&(df.diag=='6')&(df.off_diag=='-4')

        df['mat_type'] = df['mat_type'].map({'A':'equal mismatch', 'B':'linear mismatch'})

        sns.stripplot(x='mat_type', y='auc', data=df, hue='mat_type', ax=ax[i,0])
        ax[i,0].plot([2], df.loc[idx]['auc'].values, 'ok', ms=6, label='chosen score')
        ax[i,0].set_xlabel("Substitution Matrix")
        ax[i,0].set_ylabel("ROC AUC")
        ax[i,0].set_xticks(range(3))
        ax[i,0].set_xticklabels(xlbls)

        sns.scatterplot(x='actual_fpr', y='actual_tpr', data=df.loc[df['mat_type']=='linear mismatch'],
                        color='grey', ax=ax[i,1], s=10, label='linear mismatch')
        sns.scatterplot(x='actual_fpr', y='actual_tpr', data=df.loc[df['mat_type']=='equal mismatch'],
                        color=sns.color_palette()[0], ax=ax[i,1], s=10, label='equal mismatch')

        sns.scatterplot(x='actual_fpr', y='actual_tpr', data=df.loc[df['mat_type']=='equal mismatch'],
                        color='grey', ax=ax[i,2], s=10, label='equal mismatch')
        sns.scatterplot(x='actual_fpr', y='actual_tpr', data=df.loc[df['mat_type']=='linear mismatch'],
                        color=sns.color_palette()[1], ax=ax[i,2], s=10, label='linear mismatch')

        for j in [1,2]:
            sns.scatterplot(x='actual_fpr', y='actual_tpr', data=df.loc[idx],
                            color='k', ax=ax[i,j], s=20, label='chosen score')
            ax[i,j].set_xlabel("Actual False Positive Rate")
            ax[i,j].set_ylabel("Actual True Positive Rate")
            ax[i,j].legend(loc='best', frameon=False)
        ax[i,0].set_title(ttls[i], loc='left')

    for a in ax.ravel():
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
        r_all, p_all = pearsonr(np.log(frequency[idx_all]), mutability[idx_all])

        # Mode scale notes only
        idx_mode = mode_notes[np.isfinite(mutability[mode_notes]) &
                               (frequency[mode_notes] > 0)]
        ax_scat.plot(frequency[idx_mode], mutability[idx_mode], 'o',
                     color=col_mode, alpha=0.7, label=f'{mode.capitalize()} scale')
        sns.regplot(x=frequency[idx_mode], y=mutability[idx_mode],
                    logx=True, scatter=False, color=col_mode, ax=ax_scat)
        r_mode, p_mode = pearsonr(np.log(frequency[idx_mode]), mutability[idx_mode])

        # Correlation text, colour-matched to the regression lines
        ax_scat.text(0.65, 0.55,
                     f"$r$ = {r_all:5.2f}, $p$ = {p_all:5.2g}",
                     transform=ax_scat.transAxes, fontsize=10,
                     color=col_all)
        ax_scat.text(0.65, 0.45,
                     f"$r$ = {r_mode:5.2f}, $p$ = {p_mode:5.2g}",
                     transform=ax_scat.transAxes, fontsize=10,
                     color=col_mode)

        ax_scat.set_xlabel("Count")
        ax_scat.set_ylabel("Mutability")
        ax_scat.set_xscale('log')
        ax_scat.set_title(mode.capitalize())
        ax_scat.legend(loc='upper right', frameon=False)

        for a in [ax_count, ax_mut, ax_scat]:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("si2_mutability_by_mode.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si2_mutability_by_mode.pdf"), bbox_inches='tight')


#######################################################################
### SI Fig 3 :: Substitution graphs - normalization and mode separation


def plot_submat_comparison(prune=0.01, ipid=7, mode_alg='exact_pent'):
    """
    Three-tier comparison of pitch substitution graphs:
      Row A: Bronson British/American (Savage), absolute and normalized rate.
      Row B: TheSession pooled across all modes, absolute and normalized rate.
      Row C: TheSession normalized log-odds graphs, separated by mode
             (major, minor, mixolydian, dorian).

    Parameters
    ----------
    prune : float, optional
        Notes with a marginal frequency below this threshold are suppressed.
        Default is 0.01.
    ipid : int, optional
        Index into the PID-threshold axis of TheSession matrix arrays.
        Index 7 corresponds to PID = 0.85.  Default is 7.
    mode_alg : str, optional
        Mode-detection algorithm label used in the figure-data filename.
    """
    fig = plt.figure(figsize=(10,14))
    gs = GridSpec(6, 2, figure=fig, hspace=0.15, wspace=0.2, height_ratios=[1,.1,1,.1,1,1])
    ax_A = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_B = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    ax_C = [fig.add_subplot(gs[i, j]) for i in [4,5] for j in [0,1]]

    letters = np.arange(12)

    # --- Row A: Savage Bronson (absolute and normalized) ---
    mat_sav = np.load(PATH_FIG_DATA.joinpath("submat-savage_english.npy"))
    main_figs.plot_submat_graph(letters, mat_sav, ax=ax_A[0], norm=False, prune=prune)
    main_figs.plot_submat_graph(letters, mat_sav, ax=ax_A[1], norm=True, prune=prune)
    ax_A[0].set_title("Absolute Rate")
    ax_A[1].set_title("Normalized Rate")

    # --- Row B: TheSession pooled across all modes ---
    mat_all = np.load(PATH_FIG_DATA.joinpath("submat-all.npy"))[ipid]
    main_figs.plot_submat_graph(letters, mat_all, ax=ax_B[0], norm=False, prune=prune)
    main_figs.plot_submat_graph(letters, mat_all, ax=ax_B[1], norm=True, prune=prune)
    ax_B[0].set_title("Absolute Rate")
    ax_B[1].set_title("Normalized Rate")

    # --- Row C: TheSession mode-specific normalized graphs ---
    for i, (mode, scale) in enumerate(MODES.items()):
        path_mat = PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy")
        mat = np.load(path_mat)[ipid]
        main_figs.plot_submat_graph(letters, mat, ax=ax_C[i], norm=True,
                                    prune=prune, mode=scale)
        ax_C[i].set_title(mode.capitalize())

    # Row letters (A, B, C) anchored to the leftmost axes of each row
    for ax_first, lbl in zip([ax_A[0], ax_B[0], ax_C[0]], "ABC"):
        ax_first.text(-0.10, 1.05, lbl, transform=ax_first.transAxes,
                      fontsize=18, fontweight='bold', va='top')

    # Row labels on the left margin
    row_labels = [
        ("Bronson\n(Savage 2022)", ax_A[0]),
        ("TheSession\n(all modes)", ax_B[0]),
        ("TheSession\n(by mode)",   ax_C[0]),
    ]
    for lbl, ax in row_labels:
        ax.text(-0.30, 0.5, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                ha='center', va='center', rotation=90)

    fig.savefig(PATH_FIG.joinpath("si3_submat_comparison.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si3_submat_comparison.pdf"), bbox_inches='tight')



    
#######################################################################
### SI Fig 4 :: Substitution matrix correlations across PID thresholds


def plot_submat_corr(mode_alg='exact_pent', ipid=7):
    """
    Compare pairwise correlations between substitution matrices to show that
    differences between modes are larger than differences between biological
    substitution matrices at different levels of sequence divergence.

    Four panels:
      A: Heatmap of pairwise correlations between TheSession log-odds matrices
         across PID thresholds.
      B: Heatmap of pairwise correlations between mode-specific matrices
         (major, minor, mixolydian, dorian) at a fixed PID threshold.
      C: Heatmap of pairwise correlations between BLOSUM35--BLOSUM90 matrices.
      D: KDE distributions of the off-diagonal pairwise correlations for all
         three groups, illustrating that modes are more dissimilar to each other
         than BLOSUM matrices are.
    """
    pid_list = np.arange(0.5, 1.0, 0.05)
    blosum_nums = np.arange(35, 95, 5)
    mode_list = list(MODES.keys())
    pid_labels = [f"{p:.0%}" for p in pid_list]
    blosum_labels = [f"{n/100:.0%}" for n in blosum_nums]

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.subplots_adjust(wspace=0.45)

    def pairwise_corr(obs_list):
        N = len(obs_list)
        return np.array([[utils.get_corr(obs_list[i].ravel(), obs_list[j].ravel())
                          for j in range(N)] for i in range(N)])

    def add_heatmap(corr, axis, labels, title):
        N = len(corr)
        im = axis.imshow(corr, vmin=0.6, vmax=1)
        fig.colorbar(im, ax=axis)
        axis.set_xticks(range(N))
        axis.set_xticklabels(labels, rotation=90)
        axis.set_yticks(range(N))
        axis.set_yticklabels(labels)
        axis.set_title(title)
        return corr

    # --- A: Musical matrices across PID thresholds ---
    mat = np.load(PATH_FIG_DATA.joinpath("submat-all.npy"))
    obs_pid = [SM.obs_mat_to_log_odds(m) for m in mat]
    corr_pid = add_heatmap(pairwise_corr(obs_pid), ax[0], pid_labels, "Musical (PID threshold)")

    # --- B: Mode-specific matrices at fixed PID ---
    path_list = [PATH_FIG_DATA.joinpath(f"submat-{mode_alg}-{mode}.npy") for mode in mode_list]
    obs_mode = [SM.obs_mat_to_log_odds(np.load(p)[ipid]) for p in path_list]
    corr_mode = add_heatmap(pairwise_corr(obs_mode), ax[1], mode_list, "Musical (mode)")

    # --- C: BLOSUM matrices ---
    blosum_mat = [getattr(parasail, f"blosum{x}").matrix for x in blosum_nums]
    corr_blosum = add_heatmap(pairwise_corr(blosum_mat), ax[2], blosum_labels, "Biological (BLOSUM)")

    # --- D: KDE of off-diagonal correlations ---
    for corr, label in [(corr_pid, 'Musical (PID)'), (corr_mode, 'Musical (mode)'), (corr_blosum, 'Biological')]:
        mask = ~np.eye(len(corr), dtype=bool)
        sns.kdeplot(corr[mask], label=label, ax=ax[3])
    ax[3].set_xlabel("Pearson's $r$")
    ax[3].set_ylabel("Density")
    ax[3].legend(loc='best', frameon=False)
    ax[3].set_title("Pairwise correlation distributions")

    for a, lbl in zip(ax, 'ABCD'):
        a.text(-0.15, 1.07, lbl, transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    fig.savefig(PATH_FIG.joinpath("si4_submat_corr.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si4_submat_corr.pdf"), bbox_inches='tight')



#######################################################################
### SI Fig 5 :: Within-bar positional substitution rate by meter and dance type


def plot_bar_pos_sub_rate_meters_dances(ipid=7):
    """
    Within-bar positional substitution rates separated by meter (top row)
    and dance type (bottom row).

    Each panel mirrors the style of the main-text figure: bars show the
    substitution rate at each within-bar position, coloured by metrical
    hierarchy, with grey whiskers for bootstrapped 95% confidence intervals.
    METER_LIST has 5 entries and DANCE_LIST has 4; the last column of the
    dance row is left empty.

    Parameters
    ----------
    ipid : int, optional
        Index into the PID-threshold axis of the saved rate arrays.
        Index 7 corresponds to PID = 0.85.  Default is 7.
    """
    col_hier = np.array(sns.color_palette("mako", n_colors=4))

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 5, figure=fig, hspace=0.65, wspace=0.4)

    def plot_panel(ax, rate, X, width, hierarchy):
        ax.bar(X, rate[0], width, color=col_hier[hierarchy], alpha=0.7, ec='k')
        for j in range(len(X)):
            ax.plot([X[j]] * 2, [rate[2, j], rate[3, j]], '-', color='grey')
        ax.set_xticks(X)
        ax.set_xticklabels(np.arange(len(X)) + 1)
        ax.set_xlabel("Position in bar (eighth note)")
        ax.set_ylabel("Substitution rate")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # --- Row 0: Meters ---
    for i, meter in enumerate(METER_LIST):
        ax = fig.add_subplot(gs[0, i])
        rate = np.load(PATH_FIG_DATA.joinpath(f"bar_pos_rate-{meter.replace('/', '_')}.npy"))[ipid]
        sd = SUBDIV_METER[meter]
        X = np.linspace(0, 1, sd + 1)[:-1]
        width = 1 / sd / 1.5
        plot_panel(ax, rate, X, width, HIERARCHY[meter])
        ax.set_title(f"Meter: {meter}")

    # --- Row 1: Dance types ---
    for i, dance in enumerate(DANCE_LIST):
        ax = fig.add_subplot(gs[1, i])
        rate = np.load(PATH_FIG_DATA.joinpath(f"bar_pos_rate-{dance}.npy"))[ipid]
        sd = SUBDIV_DANCE[dance]
        X = np.linspace(0, 1, sd + 1)[:-1]
        width = 1 / sd / 1.5
        plot_panel(ax, rate, X, width, HIERARCHY_DANCE[dance])
        ax.set_title(dance.capitalize())

    # Group labels
    fig.text(0.5, 0.97, "By meter",      ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.50, "By dance type", ha='center', va='top', fontsize=13, fontweight='bold')

    # Shared legend for hierarchy colouring
    handles = [Patch(color=c) for c in col_hier[:3]]
    fig.legend(handles, ["Main beat", "2nd beat", "3rd beat"],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(PATH_FIG.joinpath("si5_bar_pos_sub_rate_meters_dances.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si5_bar_pos_sub_rate_meters_dances.pdf"), bbox_inches='tight')


#######################################################################
### SI Fig 6 :: Covariance and repetition for all meters


def plot_cov_rep_all_meters(nbars=8):
    """
    Position-position covariance (top row) and repetition covariance
    (bottom row) for all meters in METER_LIST.

    Each column corresponds to one meter.  Panels are produced using
    ``main_figs.plot_cov_mat`` with the precomputed
    ``part_cov-<meter>.npy`` files.

    Parameters
    ----------
    nbars : int, optional
        Number of bar tick labels to display on each axis.  Default is 8.
    """
    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.6)

    row_labels = ['Covariance', 'Repetition covariance']

    for col, meter in enumerate(METER_LIST):
        path = PATH_FIG_DATA.joinpath(f"part_cov-{meter.replace('/', '_')}.npy")
        cov, rep = np.load(path)
        for row, (mat, lbl) in enumerate(zip([cov, rep], row_labels)):
            ax = fig.add_subplot(gs[row, col])
            main_figs.plot_cov_mat(fig, ax, mat, nbars=nbars, cbar_lbl=lbl)
            if row == 0:
                ax.set_title(f"Meter: {meter}")

    fig.savefig(PATH_FIG.joinpath("si6_cov_rep_all_meters.png"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("si6_cov_rep_all_meters.pdf"), bbox_inches='tight')

