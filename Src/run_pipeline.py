from collections import Counter
from multiprocessing import Pool
import pickle
import shutil
from subprocess import Popen, PIPE
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from tqdm import tqdm

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.analysis import key_mode as KMF
from thesession.alignment import parts as PA
from thesession.structure import part_separation as PS
from thesession.io import savage_loader as savage
from thesession.io import seq_io
from thesession.analysis import substitution as SM
from thesession import utils


###################################################################################################
### Common functions


### Create folder + files and run mmseqs

### run_mmseqs will not work properly for parts!!!
### To go here or somewhere else? Probably a separate module
def run_mmseqs(df, name, go=4, ge=3, ref='ref', save_fasta=True):
    """
    Write sequences to FASTA and run an MMseqs2 all-vs-all easy-search.

    Creates a subdirectory ``PATH_MMSEQS / name``, writes a FASTA file
    of pitch-class sequences, writes a custom substitution matrix, then
    invokes ``mmseqs easy-search``.  Temporary MMseqs2 files are
    deleted on success.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with at least a ``tchroma`` column (pitch-class sequence)
        and a column named ``ref`` (or the value of ``ref``) for
        sequence identifiers.
    name : str
        Subdirectory name under ``PATH_MMSEQS`` and dataset label used
        for FASTA and result file naming.
    go : int, optional
        Gap-open penalty (positive integer passed to MMseqs2).
        Default is ``4``.
    ge : int, optional
        Gap-extend penalty (positive integer passed to MMseqs2).
        Default is ``3``.
    ref : str, optional
        Column name in ``df`` to use as sequence identifiers.
        Default is ``'ref'``.
    save_fasta : bool, optional
        If ``True`` (default), write the FASTA file before running
        MMseqs2.  Set to ``False`` if the FASTA already exists.

    Returns
    -------
    None

    Notes
    -----
    Raises an exception if ``mmseqs`` is not found on the system PATH.
    MMseqs2 output is in ``--format-mode 4`` (tab-separated with
    headers).  The substitution matrix uses ``SM.basic_submat_A(6, -4)``
    (match score 6, mismatch score -4) for the 12-pitch-class alphabet.
    """
    # First check that mmseqs is installed / in the system's path
    if isinstance(shutil.which('mmseqs'), type(None)):
        raise Exception("MMseqs is not found in the system path. Aborting!")

    # Save to fasta
    path_base = PATH_MMSEQS.joinpath(f'{name}')
    path_base.mkdir(parents=True, exist_ok=True)
    fasta_name = f'all_seq_{name}.fasta'
    if save_fasta:
        path_fasta = path_base.joinpath(fasta_name)
        seq_io.write_all_seq_to_fasta(df.tchroma, df[ref], path_fasta)

    # Save substitution matrix file
    path_submat = PATH_MMSEQS.joinpath(f'{name}/matrix.out')
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

    if pipe_output.returncode == 0:
        print("MMseqs has completed successfully!")
        shutil.rmtree(path_base.joinpath("tmp"))
    else:
        print("An error has occurred while running MMseqs!")
        print(stderr.decode())


### Load mmseqs results (or run mmseqs if not done yet)
def load_mmseqs(df, dataset, ref='setting_id', redo=False, annotate=True, save_fasta=True):
    """
    Load MMseqs2 results from cache, running MMseqs2 first if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table passed to ``run_mmseqs`` (if run) and to
        ``seq_io.load_mmseqs_pairwise`` (if ``annotate=True``).
    dataset : str
        Dataset name; determines subdirectory and result file path.
    ref : str, optional
        Sequence-identifier column passed to ``run_mmseqs``.
        Default is ``'setting_id'``.
    redo : bool, optional
        If ``True``, re-run MMseqs2 even if a result file exists.
        Default is ``False``.
    annotate : bool, optional
        Passed to ``seq_io.load_mmseqs_pairwise``; if ``True`` (default),
        adds an ``in_fam`` column.
    save_fasta : bool, optional
        Passed to ``run_mmseqs``; default is ``True``.

    Returns
    -------
    pandas.DataFrame
        Filtered and optionally annotated MMseqs2 result table.
    """
    path = PATH_MMSEQS.joinpath(f"{dataset}/result.m8")
    if not path.exists() or redo:
        print(f"Running mmseqs for '{dataset}'...")
        run_mmseqs(df, dataset, ref=ref, save_fasta=save_fasta)
    else:
        print(f"Loading cached mmseqs results for '{dataset}'")
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
def data_for_fig1(redo=True):
    """
    Produce ROC-curve data for Figure 1 (tune-family identification).

    Runs MMseqs2 all-vs-all alignment on three datasets — TheSession,
    Meertens MTC-FS-INST-2.0, and Savage et al. (English) — using the
    12-pitch-class letter encoding.  For each dataset, computes ROC
    curve and AUC treating within-family pairs as positives and
    cross-family pairs as negatives.  Results are pickled to
    ``PATH_FIG_DATA / "fig1_roc_curve_data.pkl"``.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, reprocess all datasets from scratch.
        Default is ``False``.

    Returns
    -------
    None

    Notes
    -----
    The TheSession run uses the full (unfiltered) dataset to maximise
    the number of within-family positives for the ROC curve, then
    applies a post-hoc filter for grace notes, polyphony, and multiple
    voices (but not repeat consistency).
    """
    # Create container for data for figures
    fig_data = {}

    ### TheSession
    print("Running on TheSession data")

    # Load the full TheSession dataset
    path = PATH_CACHE.joinpath("thesession_tunes.pkl")
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
    print(f"  AUC: {fig_data[f'{dataset}_auc']:.3f}")

    ### Meertens
    print("Running on Meertens data")
    df = load_tunes.load_meertens_data(redo=redo)[0]

    dataset = "meertens"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, "ref", redo=redo), dataset, fig_data)
    fig_data = get_total_positives(df, dataset, 'song_id', fig_data)
    print(f"  AUC: {fig_data[f'{dataset}_auc']:.3f}")

    ### Savage et al.
    print("Running on data from Savage et al.")
    df = savage.load_savage_df(full=True, redo=redo)
    df = df.loc[df.Language=='English']

    dataset = "savage_english"
    fig_data = get_roc_and_auc(load_mmseqs(df, dataset, "ref", redo=redo), dataset, fig_data)
    fig_data = get_total_positives(df, dataset, 'chapter', fig_data)
    print(f"  AUC: {fig_data[f'{dataset}_auc']:.3f}")

    path = PATH_FIG_DATA.joinpath("fig1_roc_curve_data.pkl")
    pickle.dump(fig_data, open(path, 'wb'))
    print(f"Saved figure 1 data to {path}")


### Get overall tpr/fpr, accounting for screening stage
def get_total_positives(df, dataset, x='tune_id', fig_data=None):
    """
    Count the total number of positive and negative pairs in a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with a column ``x`` containing family identifiers.
    dataset : str
        Dataset label used as a key prefix in ``fig_data``.
    x : str, optional
        Column name for family membership.  Default is ``'tune_id'``.
    fig_data : dict or None, optional
        Existing figure-data dict to update.  A new dict is created if
        ``None``.  Default is ``None``.

    Returns
    -------
    fig_data : dict
        Updated dict with keys ``'{dataset}_positives'``,
        ``'{dataset}_negatives'``, and ``'{dataset}_total'``.

    Notes
    -----
    Positives are the number of within-family unordered pairs,
    computed as ``sum(n*(n-1)/2)`` for each family of size ``n``.
    Total is ``len(df)^2`` (all ordered pairs including self-pairs).
    """
    total = len(df)**2
    # Within-family unordered pairs: n*(n-1)/2 per family
    positives = np.sum([n * (n - 1) / 2 for n in df[x].value_counts().values])
    negatives = total - positives
    fig_data[f'{dataset}_positives'] = positives
    fig_data[f'{dataset}_negatives'] = negatives
    fig_data[f'{dataset}_total'] = total
    return fig_data


# Get roc and roc-auc
def get_roc_and_auc(res, dataset, fig_data=None):
    """
    Compute ROC curve and AUC for a set of MMseqs2 alignment hits.

    Parameters
    ----------
    res : pandas.DataFrame
        Annotated hit table with columns ``in_fam`` (bool, True for
        within-family pairs) and ``fident`` (fractional sequence
        identity used as the score).
    dataset : str
        Dataset label used as a key prefix in ``fig_data``.
    fig_data : dict or None, optional
        Existing figure-data dict to update.  A new dict is created if
        ``None``.  Default is ``None``.

    Returns
    -------
    fig_data : dict
        Updated with keys ``'{dataset}_roc'`` (list of [fpr, tpr]),
        ``'{dataset}_auc'``, ``'{dataset}_screened'``,
        ``'{dataset}_screened_positives'``, and
        ``'{dataset}_screened_negatives'``.
    """
    if fig_data is None:
        fig_data = {}
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
    """
    Extract ``(tune_id, part_index)`` from a part_id string.

    Parameters
    ----------
    s : str
        Part ID string in the format ``"{tune_id}_{setting_id}_{part_no}"``.

    Returns
    -------
    tuple of int
        ``(tune_id, part_no)`` — the first and last underscore-delimited
        fields as integers.

    Notes
    -----
    This is used to group hits by (tune, part) regardless of which
    setting they come from.
    """
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
    """
    Run the full TheSession part-alignment pipeline.

    This is the central computation step for Figures 2–5.  It:

    1. Loads the cleaned TheSession dataset (``~2 h`` first run).
    2. Splits each setting into structural parts.
    3. Writes parts to FASTA and runs MMseqs2.
    4. Prunes hits from duplicate parts.
    5. Annotates and filters all surviving hit pairs.

    Parameters
    ----------
    redo : bool, optional
        Passed to every sub-step; if ``True``, all caches are
        invalidated and recomputed.  Default is ``False``.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned setting-level metadata.
    tunes : dict
        Music21-parsed feature dicts keyed by ``setting_id``.
    df_parts : pandas.DataFrame
        Part-level metadata (one row per extracted part).
    parts_data : dict
        Part feature dicts keyed by ``part_id``.
    res : pandas.DataFrame
        Full annotated MMseqs2 hit table (all surviving pairs).
    res0 : pandas.DataFrame
        Filtered hit table (equal duration, equal meter, 0.5 < fident < 1).
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (substitutions, matches, etc.).
    """
    # Load a cleaned dataset
    # (code takes about 2 hours to run)
    print(f"Loading data (redo is {'on' if redo else 'off'})")
    t0 = time.time()
    df, tunes = load_tunes.load_thesession_data(redo=redo)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Extract parts
    print(f"Splitting tunes into parts")
    t0 = time.time()
    df_parts, parts_data = PS.get_all_parts_thesession(df, tunes, redo=redo)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Write parts to fasta
    print(f"Writing to fasta")
    seq_io.write_parts_thesession(parts_data)

    # (Run and ) load mmseqs2 results
    ### CHEKC THIS!!! PROBABLY DOES NOT WORK!!!
    t0 = time.time()
    res = load_mmseqs(parts_data, "thesession_parts", redo=redo, annotate=False, save_fasta=False)
    print(f"  mmseqs gave {len(res)} tune pairs ({time.time()-t0:.1f}s)")

    res = PA.prune_identical_parts(res, parts_data)
    print(f"Pruning identical parts leaves {len(res)} tune pairs")

    # Align parts using new algorithm
    print("Annotating alignments...")
    t0 = time.time()
    res, res0, mismatches = PA.annotate_res(df, df_parts, res, parts_data, redo=redo)
    print(f"  Final set: {len(res0)} tune pairs ({time.time()-t0:.1f}s)")

    return df, tunes, df_parts, parts_data, res, res0, mismatches



# Run analyses for Fig 2:
#    Note prevalence, mutability, key finding, IDyOM
def data_for_fig2(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False):
    """
    Compute and save all data needed for Figure 2.

    Runs three analyses:

    1. Note prevalence and mutability substitution matrices
       (``note_prevalence_mutability``).
    2. Savage et al. substitution matrix
       (``note_prevalence_mutability_savage``).
    3. Key-finding accuracy by note stability
       (``note_stability_key_finding``).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned setting-level metadata.
    tunes : dict
        Music21-parsed feature dicts keyed by ``setting_id``.
    df_parts : pandas.DataFrame
        Part-level metadata.
    parts_data : dict
        Part feature dicts keyed by ``part_id``.
    res : pandas.DataFrame
        Full annotated hit table.
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    redo : bool, optional
        If ``True``, recompute cached results.  Default is ``False``.

    Returns
    -------
    None
    """
    print("Computing substitution matrices...")
    _ = note_prevalence_mutability(res0, mismatches, tunes, redo=redo)
    print("Computing Savage substitution matrix...")
    _ = note_prevalence_mutability_savage(redo=redo)
    print("Computing key-finding accuracy...")
    _ = note_stability_key_finding(df, tunes, res0, parts_data, 0.85, redo=redo)


### Get substitution matrices for different PID thresholds
def get_submat_by_pid(res0, mismatches, pid_list, path_mat, alpha=0.5, redo=False):
    """
    Compute or load substitution matrices across a range of PID thresholds.

    For each threshold in ``pid_list``, filters ``res0`` to hits with
    ``fident > pid``, accumulates the pitch-class observation table via
    ``PA.subs_to_observations``, converts to a normalised matrix via
    ``SM.convert_observations_to_matrix``, and stacks the results.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (same index as ``res0``).
    pid_list : array-like of float
        Sequence of PID thresholds to evaluate.
    path_mat : pathlib.Path
        Path for caching the result as a ``.npy`` file.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        If ``True``, recompute even if the cache exists.  Default is
        ``False``.

    Returns
    -------
    mat : numpy.ndarray, shape (len(pid_list), 12, 12)
        Normalised substitution matrix for each PID threshold.
    """
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
    """
    Compute substitution matrices for all mode, dance, and combined
    groupings across a range of PID thresholds.

    Matrices are computed for:

    * All pairs combined.
    * Each of the four modes under four compatibility algorithms
      (``'exact'``, ``'loose'``, ``'exact_pent'``, ``'loose_pent'``).
    * Each dance type.
    * Each (mode, dance) combination under ``'exact_pent'``.

    Results are cached as ``.npy`` files under ``PATH_FIG_DATA``.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    tunes : dict
        Full tune dict (used only for melodic interval distribution).
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        If ``True``, recompute cached matrices.  Default is ``False``.

    Returns
    -------
    mat_dict : dict
        Keys are grouping labels (e.g. ``'all'``, ``'exact-major'``,
        ``'reel'``, ``'major-reel'``); values are arrays of shape
        ``(len(pid_list), 12, 12)``.
    """

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
        idx = np.array((res0.target_dance==dance)&(res0.query_dance==dance), bool)
        mat_dict[k] = get_submat_by_pid(res0.loc[idx], mismatches.loc[idx], pid_list, path_mat, alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {k}")

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
    """
    Compute or load the substitution matrix for Savage et al. (English).

    Parameters
    ----------
    redo : bool, optional
        If ``True``, recompute even if the cache exists.  Default is
        ``False``.

    Returns
    -------
    numpy.ndarray, shape (12, 12)
        Normalised substitution matrix for the Savage English corpus.
    """
    path = PATH_FIG_DATA.joinpath(f"submat-savage_english.npy")
    if path.exists() and not redo:
        print("  Loading cached Savage substitution matrix")
        return np.load(path)
    else:
        print("  Computing Savage substitution matrix...")
        df = savage.load_savage_df(full=True, redo=False)
        df = df.loc[df.Language=='English']
        obs, letters, mat = savage.get_submat(df.loc[df.Language=='English'])
        np.save(path, mat)
        return mat


### Estimate melody key, using different sets of notes:
### original note order, the most conserved notes, and the least conserved notes
def _key_finding_init(res, parts, profiles):
    """
    Initialise per-process globals for the key-finding multiprocessing pool.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table shared across all workers.
    parts : dict
        Parts dict shared across all workers.
    profiles : dict
        Modal profiles dict shared across all workers.

    Returns
    -------
    None
    """
    global _kf_res, _kf_parts, _kf_profiles
    _kf_res, _kf_parts, _kf_profiles = res, parts, profiles


def _key_finding_worker(args):
    """
    Worker function for parallel key-finding evaluation.

    Parameters
    ----------
    args : tuple
        ``(tune_id, p0, meter)`` — tune identifier, part index (0-based),
        and meter string.

    Returns
    -------
    result
        Return value of ``KMF.predict_key_family`` for this tune/part.
    """
    tune_id, p0, meter = args
    return KMF.predict_key_family(_kf_res, _kf_parts, _kf_profiles,
                                  tune_id, p0, meter, factor=4, pid=0.5, nran=10)


def note_stability_key_finding(df, tunes, res0, parts_data, pid=0.85, redo=False):
    """
    Evaluate key-finding accuracy using note stability as a feature.

    For each tune/part pair that appears in at least 10 within-family
    hit pairs, runs ``KMF.predict_key_family`` using modal profiles
    inferred from the corpus.  Results are parallelised across
    ``N_PROC`` processes and cached to ``PATH_FIG_DATA``.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned metadata with ``tune_id`` and ``meter`` columns.
    tunes : dict
        Music21-parsed tune feature dicts.
    res0 : pandas.DataFrame
        Filtered hit table.
    parts_data : dict
        Parts feature dicts.
    pid : float, optional
        Minimum fractional identity threshold for including pairs.
        Default is ``0.85``.
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    correct_key : numpy.ndarray
        Array of key-finding results (one entry per candidate tune/part).
        The exact content depends on ``KMF.predict_key_family``.

    Notes
    -----
    Only tune/part pairs with at least 10 hit-table appearances are
    evaluated to ensure sufficient statistical power for the MSA-based
    stability calculation.
    """
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
        args = [(t, p, m) for (t, p), m in zip(part_set, meter_list)]
        with Pool(N_PROC, initializer=_key_finding_init,
                  initargs=(res0, parts_data, mode_profiles)) as pool:
            correct_key = list(tqdm(pool.imap(_key_finding_worker, args), total=len(part_set)))
        correct_key = np.array(correct_key)
        np.save(path, correct_key)
        return correct_key



###################################################################################################
### Note substitutions (Fig. 3)

    # Run analyses for Fig 3:
    #    Substitution rates + log odds, sub distance (separate by mode, dance, mode and dance, all PID)
def data_for_fig3(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False, mode_alg='exact_pent'):
    """
    Compute and save all data needed for Figure 3 (substitution analysis).

    Runs:

    1. Substitution matrices for all groupings
       (``note_prevalence_mutability``).
    2. Melodic interval distribution (``mint_dist``).
    3. Empirical substitution distance distributions for all pairs and
       per mode (``note_sub_dist``).
    4. Substitution distance for the Savage corpus
       (``note_sub_dist_savage``).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned setting metadata.
    tunes : dict
        Music21-parsed feature dicts.
    df_parts : pandas.DataFrame
        Part-level metadata.
    parts_data : dict
        Part feature dicts.
    res : pandas.DataFrame
        Full annotated hit table.
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.
    mode_alg : str, optional
        Mode-matching algorithm for per-mode breakdowns.
        Default is ``'exact_pent'``.

    Returns
    -------
    None
    """
    print("Computing substitution matrices...")
    _ = note_prevalence_mutability(res0, mismatches, tunes, redo=redo)
    print("Computing melodic interval distribution...")
    _ = mint_dist(tunes, redo=redo)

    path = PATH_FIG_DATA.joinpath(f"sub_dist_all.npy")
    _ = note_sub_dist(res0, mismatches, parts_data, path, alpha=0.5, redo=redo)
    note_sub_dist_savage(redo=redo)

    idx_list = utils.get_mode_indices(res0, mismatches, mode_alg)
    for mode, idx in zip(MODES.keys(), idx_list):
        path = PATH_FIG_DATA.joinpath(f"sub_dist_{mode_alg}_{mode}.npy")
        _ = note_sub_dist(res0.loc[idx], mismatches.loc[idx], parts_data, path, alpha=0.5, redo=redo)



### Get the melodic interval distribution
def mint_dist(tunes, redo=False):
    """
    Compute or load the normalised melodic interval distribution.

    Parameters
    ----------
    tunes : dict
        Mapping from ``setting_id`` to tune feature dict.
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Normalised interval distribution array as returned by
        ``PA.get_mint_dist``.
    """
    path = PATH_FIG_DATA.joinpath(f"mint_dist_tunes.npy")
    if path.exists() and not redo:
        print("  Loading cached melodic interval distribution")
        return np.load(path)
    else:
        print("  Computing melodic interval distribution...")
        mint = PA.get_mint_dist(tunes)
        np.save(path, mint)
        return mint

### Expected substitution distance rate.
### For each tune part, get the expected melodic interval
### distribution one would obtain by shuffling repeatedly.
def get_base_sub_dist_rate(res, mismatches, parts, alpha=0.5):
    """
    Compute the expected substitution distance distribution under a
    null model of random note pairing within each part.

    For each part that appears in the hit table, all pairwise
    absolute pitch differences are tallied (weighted by the number of
    pairs the part is involved in).  This gives the background
    distribution of interval sizes one would expect if substitutions
    were drawn uniformly.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (used for indexing; content not
        used here).
    parts : dict
        Parts feature dicts keyed by ``part_id``.
    alpha : float, optional
        Inverse-frequency weighting exponent (not currently applied;
        included for API consistency).  Default is ``0.5``.

    Returns
    -------
    tot : numpy.ndarray of float, shape (13,)
        Raw count of each interval distance from 1 to 13 semitones
        (index 0 = 1 semitone, index 12 = 13 semitones).
    """
    M = np.arange(1, 14)
    tot = np.zeros(M.size, float)
    parts_dict = Counter(res[['query', 'target']].values.ravel())
    # As far as I can see, the weights do not need to be applied here! ***CHECK
#   weights = utils.inverse_frequency_weights(res, alpha)
    for p, c in parts_dict.items():
        tmidi = parts[p][0][1].astype(int)
        # Get the difference of all notes with all notes, to get
        # all possible melodic intervals
        count = Counter(np.abs(tmidi[:,None] - tmidi[None,:]).ravel())
        # Should this not also be normalized by melody length???
        for k, v in count.items():
            if k in M:
                tot[k-1] += v * c# * w
    return tot


### Empirical substitution distance rate
def note_sub_dist(res, mismatches, parts, path, alpha=0.5, redo=False):
    """
    Compute the empirical substitution distance distribution and its
    log-odds relative to the null model, across PID thresholds.

    For each PID threshold in ``np.arange(0.5, 1, 0.05)``, computes the
    weighted count of observed substitution distances and the log-odds
    ratio against the null (random-pairing) expectation.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (same index as ``res``).
    parts : dict
        Parts feature dicts keyed by ``part_id``.
    path : pathlib.Path
        Cache file path for the result array.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    out : numpy.ndarray, shape (len(pid_list), 2, 13)
        ``out[i, 0]`` is the weighted count of substitution distances at
        PID threshold ``pid_list[i]``; ``out[i, 1]`` is the log-odds
        ratio against the null model.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        pid_list = np.arange(0.5, 1, 0.05)
        X = np.arange(1, 14)
        out = []
        for pid in tqdm(pid_list, desc="  pid thresholds"):
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
            # log(observed fraction) - log(expected fraction)
            Y2 = np.log(Y / np.sum(Y)) - np.log(tot / tot.sum())

            out.append([Y, Y2])
        out = np.array(out)
        np.save(path, out)
        return out


### Empirical substitution distance rate for the Bronson (British/American)
### and Japanese collections
def note_sub_dist_savage(redo=False):
    """
    Compute or load substitution distance distributions for the Savage
    English and Japanese corpora.

    Parameters
    ----------
    redo : bool, optional
        If ``True``, recompute even if cached.  Default is ``False``.

    Returns
    -------
    out : numpy.ndarray, shape (2, 2, 13)
        ``out[0]`` is for English, ``out[1]`` for Japanese.
        Each sub-array has ``[observed_counts, expected_counts]``.
    """
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
    """
    Compute the observed and expected substitution distance distributions
    from manually aligned Savage et al. pairs.

    Each ``PairNo`` in the DataFrame corresponds to a manually aligned
    pair of tune variants.  The function tallies absolute pitch
    differences at substitution positions and at all pairwise note
    positions (for the null expectation).

    Parameters
    ----------
    df : pandas.DataFrame
        Savage subset for one language, with columns ``PairNo``,
        ``tchroma`` (pitch-class array), and ``seq_aligned`` (aligned
        letter string, with ``'-'`` for gaps).
    redo : bool, optional
        Not used; kept for API consistency.

    Returns
    -------
    tot : numpy.ndarray of float, shape (13,)
        Counts of observed substitution distances (1–13 semitones).
    expected : numpy.ndarray of float, shape (13,)
        Counts of all pairwise distances (null model for expected
        distribution).
    """
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
        # Keep only positions where the corresponding alignment column is not a gap
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
        # Compute all pairwise absolute distances within each sequence for the null
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
    """
    Compute and save all data needed for Figure 4 (positional analysis).

    Runs:

    1. Bar-level substitution rates by meter, dance, and mode
       (``bar_rate``).
    2. Within-bar positional substitution rates by meter and dance
       (``bar_pos_rate``).
    3. Onset histograms by meter (``onset_histograms``).
    4. Correlations between positional substitution rate and metrical
       hierarchy / onset stability (``bar_pos_rate_corr``).

    Parameters
    ----------
    df : pandas.DataFrame
        Setting metadata.
    tunes : dict
        Music21-parsed feature dicts.
    df_parts : pandas.DataFrame
        Part-level metadata.
    parts_data : dict
        Part feature dicts.
    res : pandas.DataFrame
        Full annotated hit table.
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    None
    """
    print("Computing bar substitution rates...")
    _ = bar_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=redo)
    print("Computing bar position substitution rates...")
    _ = bar_pos_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=redo)
    print("Computing onset histograms...")
    _ = onset_histograms(res0, parts_data, redo=redo)
    print("Computing bar position rate correlations...")
    _ = bar_pos_rate_corr(redo=redo)


### Calculate the average substitution rate in a measure, as a function
### of the position of the measure in the part.
### For many different groups of tune parts.
def bar_rate(res0, mismatches, mode_alg='exact_pent', alpha=0.5, redo=False):
    """
    Compute per-bar substitution rates for all standard groupings.

    Restricts to the four main dance types (reel, jig, polka, hornpipe)
    which are known to have the 8-bar part structure, then computes
    bar-level substitution rates for:

    * All pairs combined.
    * Each of the three most common meters (4/4, 6/8, 2/4).
    * Each dance type.
    * Each mode (under ``mode_alg``).

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    mode_alg : str, optional
        Mode-compatibility algorithm.  Default is ``'exact_pent'``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    rate_dict : dict
        Keys are grouping labels (``'all'``, meter strings, dance names,
        mode names); values are arrays of shape
        ``(len(pid_list), 4, nbars)`` — mean, std, and 95% CI bounds.
    """
    pid_list = np.arange(0.5, 1, 0.05)
    dance_list = ['reel', 'jig', 'polka', 'hornpipe']
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
        rate_dict[dance] = get_bar_rate(res0.loc[idx], mismatches.loc[idx], pid_list, path, alpha=alpha, redo=redo)
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
    """
    Compute mean bar substitution rate with bootstrapped confidence
    intervals across PID thresholds.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table subset.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (same index as ``res``).
    pid_list : array-like of float
        PID thresholds to evaluate.
    path : pathlib.Path
        Cache file path.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    rate_stats : numpy.ndarray, shape (len(pid_list), 4, nbars)
        For each PID threshold: ``[mean, std, 2.5th_percentile,
        97.5th_percentile]`` of the per-bar substitution rate across
        bootstrap resamples.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        ci = [0.025, 0.975]
        rate_stats = []
        for pid in tqdm(pid_list, desc="  pid thresholds"):
            idx = np.array(res.fident > pid, bool)
            weights = utils.inverse_frequency_weights(res.loc[idx], alpha)
            sub_rate_all = get_bar_subrate(mismatches.loc[idx], max_bar=8)
            Y, Ysample = get_bar_subrate_stats(sub_rate_all, weights)
            Ys = np.std(Ysample, axis=0)
            # Save the mean, standard deviation, and the 95% CI
            rate_stats.append([Y, Ys] + list(np.quantile(Ysample, ci, axis=0)))
        rate_stats = np.array(rate_stats)
        np.save(path, rate_stats)
        return rate_stats


### Get bar substitution rate
def get_bar_subrate(mismatches, max_bar=8):
    """
    Compute the per-bar substitution rate for each pair in a mismatch
    table.

    Rates are expressed in bar-fraction units (substitutions per total
    bar duration) rather than raw counts, so that pairs with finer grid
    resolutions are not artificially inflated.

    Parameters
    ----------
    mismatches : pandas.DataFrame
        Sub-table of per-pair alignment statistics with columns
        ``'sub_bar'`` and ``'grid_per_bar'``.
    max_bar : int, optional
        Number of bars to include (bars beyond this index are ignored).
        Default is ``8``.

    Returns
    -------
    sub_rate_all : numpy.ndarray, shape (n_pairs, max_bar)
        Per-bar substitution rate for each pair.
    """
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
    """
    Compute within-bar positional substitution rates for all meter and
    dance groupings.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    mode_alg : str, optional
        Mode-compatibility algorithm.  Default is ``'exact_pent'``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    rate_dict : dict
        Keys are meter or dance label strings; values are arrays of
        shape ``(len(pid_list), 4, subdivision)`` returned by
        ``get_bar_pos_rate``.
    """
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
        rate_dict[dance] = get_bar_pos_rate(res0.loc[idx], mismatches.loc[idx], sd, pid_list, path, alpha=alpha, redo=redo)
        print(f"{np.sum(idx)} pairs used for {dance}")

    return rate_dict


### Calculate the substitution rate as a function of the position within a measure.
### For different PID thresholds.
def get_bar_pos_rate(res, mismatches, subdivision, pid_list, path, alpha=0.5, redo=False):
    """
    Compute within-bar positional substitution rates with bootstrapped
    confidence intervals across PID thresholds.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table subset.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics.
    subdivision : int
        Number of positions per bar (from ``SUBDIV_METER`` or
        ``SUBDIV_DANCE``).
    pid_list : array-like of float
        PID thresholds to evaluate.
    path : pathlib.Path
        Cache file path.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    numpy.ndarray, shape (len(pid_list), 4, subdivision)
        For each PID threshold: ``[mean, std, 2.5th_percentile,
        97.5th_percentile]`` of the positional substitution rate.

    Notes
    -----
    ``sub_pos`` values are in units of bar fraction [0, 1).  They are
    converted to bar-subdivision units by multiplying by
    ``subdivision`` before binning.  Rounding handles floating-point
    precision errors.
    """
    if path.exists() and not redo:
        return np.load(path)
    else:
        ci = [0.025, 0.975]
        rate_stats = []
        for pid in tqdm(pid_list, desc="  pid thresholds"):
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
    """
    Compute the weighted mean and bootstrapped distribution of
    per-bar substitution rates.

    Parameters
    ----------
    sub_rate_all : numpy.ndarray, shape (n_pairs, nbars)
        Per-pair per-bar substitution rates.
    weights : numpy.ndarray of float, shape (n_pairs,)
        Inverse-frequency weights for each pair.
    nrep : int, optional
        Number of bootstrap replicates.  Default is ``1000``.

    Returns
    -------
    Y : numpy.ndarray of float, shape (nbars,)
        Weighted mean substitution rate per bar.
    Ysample : numpy.ndarray of float, shape (nrep, nbars)
        Bootstrap distribution of the weighted mean.

    Notes
    -----
    Bootstrap resamples are drawn with replacement at the pair level.
    The weights of the resampled pairs are renormalised within each
    replicate so that the weighted mean is unbiased.
    """
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
    """
    Compute onset-position histograms (probability of a note onset at
    each within-bar position) for each meter.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table (used to group by meter).
    parts_data : dict
        Parts feature dicts keyed by ``part_id``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    hist_stats : dict
        Keys are meter strings; values are lists
        ``[mean, std, lower_CI, upper_CI]`` of shape
        ``(subdivision,)`` each.
    """
    path = PATH_FIG_DATA.joinpath("onset_histograms.pkl")
    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        ci = [0.025, 0.975]
        hist_stats = {}
        for meter in METER_LIST:
            idx = (res0.target_meter==meter)&(res0.query_meter==meter)
            hist = get_onset_histograms(res0.loc[idx], parts_data, SUBDIV_METER[meter])
            # Normalise each row to a probability distribution
            hist = hist / hist.sum(axis=1)[:,None]
            Ym = np.mean(hist, axis=0)
            Ys = np.std(hist, axis=0)
            Ylo, Yhi = np.quantile(hist, ci, axis=0)
            hist_stats[meter] = [Ym, Ys, Ylo, Yhi]
        pickle.dump(hist_stats, open(path, 'wb'))
        return hist_stats


### Multiply duration values by a factor of 2 to get units of eighth notes
def get_onset_histograms(res, parts_data, subdivision, factor=2):
    """
    Tally onset positions within a bar for all parts in a set of pairs.

    Onsets are computed from cumulative note durations and reduced to
    within-bar positions (modulo ``subdivision``) in units of eighth
    notes (``factor=2`` converts quarter-note units).

    Parameters
    ----------
    res : pandas.DataFrame
        Hit table subset filtered to one meter.
    parts_data : dict
        Parts feature dicts keyed by ``part_id``.
    subdivision : int
        Number of grid positions per bar for the given meter.
    factor : int, optional
        Multiplier to convert quarter-note durations to the desired
        grid resolution.  Default is ``2`` (eighth notes).

    Returns
    -------
    numpy.ndarray, shape (n_pairs, subdivision)
        Per-pair onset count at each within-bar position.
    """
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
    """
    Compute R² between within-bar substitution rates and metrical
    hierarchy / onset stability, across PID thresholds.

    For each PID threshold, builds a DataFrame via
    ``load_hierarchy_stability_df`` and computes:

    1. Pearson R² between metrical hierarchy and relative substitution
       rate.
    2. Pearson R² between relative onset stability and relative
       substitution rate.
    3. R² from an OLS regression of substitution rate on (hierarchy,
       end_pos).

    Parameters
    ----------
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    corr : numpy.ndarray, shape (len(pid_list), 3)
        Three R² values for each PID threshold.
    """
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

            # OLS regression of substitution rate on hierarchy + end_pos indicator
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
    """
    Build a DataFrame combining metrical hierarchy, onset stability, and
    positional substitution rate for a given PID threshold index.

    Parameters
    ----------
    ipid : int, optional
        Index into ``np.arange(0.5, 1, 0.05)`` selecting the PID
        threshold.  Default is ``7`` (corresponds to PID = 0.85).

    Returns
    -------
    pandas.DataFrame
        One row per (meter, within-bar position) combination, with
        columns ``'hierarchy'``, ``'end_pos'``, ``'meter'``,
        ``'stability'``, ``'rel_stability'`` (stability / mean_stability),
        ``'sub_rate'``, and ``'rel_sub_rate'`` (sub_rate / mean_sub_rate).
    """
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
def data_for_fig5(res0, parts_data, redo=False):
    """
    Compute and save all data needed for Figure 5 (covariance analysis).

    Calls ``part_covariance`` to compute position-position covariance
    matrices for each meter and for the 10 most-represented individual
    tune parts.

    Parameters
    ----------
    res0 : pandas.DataFrame
        Filtered hit table.
    parts_data : dict
        Parts feature dicts.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    None
    """
    print("Computing covariance matrices...")
    part_covariance(res0, parts_data, alpha=0.5, redo=redo)


### Get the covariance matrices for sets of tunes grouped by meter,
### and for a few tune families
def part_covariance(res, parts_data, factor0=2, nbars=8, alpha=0.5, redo=False):
    """
    Compute position-position covariance matrices for meter groups and
    the 10 most-represented individual tune parts.

    For each meter, computes the covariance across all pairs with the
    same meter via ``part_covariance_meter``.  Then identifies the top 10
    tune/part combinations by hit count and computes their individual
    covariance matrices.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table.
    parts_data : dict
        Parts feature dicts.
    factor0 : int, optional
        Grid factor used to convert note durations (default ``2`` for
        eighth-note resolution).
    nbars : int, optional
        Number of bars to include in each sequence window.  Default ``8``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute caches if ``True``.  Default is ``False``.

    Returns
    -------
    None
    """
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
    """
    Compute the position-position covariance and repetition matrices for
    a set of pairs with a given meter.

    For each pair, sequences are placed on a uniform eighth-note grid,
    octave-corrected, and truncated to ``ngrid = nbars * grid_per_bar``
    positions.  The covariance matrix reflects how often positions
    change together, and the repetition matrix identifies positions
    where both sequences share a note but that note differs between the
    two sequences (evidence of preserved repetition under mutation).

    Parameters
    ----------
    res : pandas.DataFrame
        Hit table subset for the desired meter/tune group.
    parts_data : dict
        Parts feature dicts.
    meter : str
        Meter string (e.g. ``'4/4'``).
    path : pathlib.Path
        Cache file path (saved as ``.npy`` with two arrays: ``[cov, rep]``).
    factor0 : int, optional
        Base grid factor (2 = eighth-note resolution).  Default is ``2``.
    nbars : int, optional
        Number of bars per window.  Default is ``8``.
    alpha : float, optional
        Inverse-frequency weighting exponent.  Default is ``0.5``.
    redo : bool, optional
        Recompute if ``True``.  Default is ``False``.

    Returns
    -------
    cov : numpy.ndarray, shape (ngrid, ngrid)
        Weighted position-position covariance matrix.
    rep : numpy.ndarray, shape (ngrid, ngrid)
        Weighted mean repetition indicator matrix.

    Notes
    -----
    If a pair's grid factor is larger than ``factor0`` (because it
    contains notes shorter than an eighth note), the sequence is
    sub-sampled at every ``ratio``-th position to restore the eighth-note
    grid.  Pairs whose sequence is shorter than ``ngrid`` after gridding
    are skipped.
    """
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

        cov = np.cov(np.array(eq).T, aweights=weights[idx])
        rep = np.average(changes, weights=weights[idx], axis=0)
        np.save(path, [cov, rep])
        return cov, rep


### Code for recreating the analyses in the paper.
### Creates and saves data for figures.
def main(redo=False):
    """
    Run the complete analysis pipeline for all figures.

    Calls each figure-data function in sequence:

    * Figure 1: tune-family identification ROC curves.
    * Main alignments: shared prerequisite for Figures 2–5.
    * Figure 2: note prevalence, mutability, key-finding.
    * Figure 3: note substitution distances.
    * Figure 4: sequence-position effects.
    * Figure 5: sequence covariance.

    Parameters
    ----------
    redo : bool, optional
        Passed to every sub-function.  If ``True``, all caches are
        invalidated and recomputed.  Default is ``False``.

    Returns
    -------
    None
    """
    print("=== Fig 1: Identifying similar tunes ===")
    data_for_fig1(redo=redo)

    print("\n=== Running main alignments ===")
    df, tunes, df_parts, parts_data, res, res0, mismatches = run_main_alignments(redo=redo)

    print("\n=== Fig 2: Note prevalence, mutability, key-finding ===")
    data_for_fig2(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=redo)

    print("\n=== Fig 3: Note substitutions ===")
    data_for_fig3(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=redo)

    print("\n=== Fig 4: Sequence position effects ===")
    data_for_fig4(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=redo)

    print("\n=== Fig 5: Sequence covariance ===")
    data_for_fig5(res0, parts_data, redo=redo)


if __name__ == "__main__":

    main(redo=False)
