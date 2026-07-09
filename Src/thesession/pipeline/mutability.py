"""Note prevalence, mutability, and key-finding.

Feeds published Fig. 2 (note mutability, substitution-matrix heatmap), Fig. 3
(key-finding accuracy), and SI2 (mutability by mode).
"""

from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from thesession.config import MODES, N_PROC, PATH_FIG_DATA
from thesession.io import tune_loader as load_tunes
from thesession.analysis import key_mode as KMF
from thesession.alignment import parts as PA
from thesession.io import savage_loader as savage
from thesession.analysis import substitution as SM
from thesession import utils
from thesession.pipeline.identification import get_uniq


###################################################################################################
### Note prevalence, mutability and key-finding
### (The shared alignment step run_main_alignments now lives in pipeline/mmseqs.py.)
def data_for_mutability(df, tunes, df_parts, parts_data, res, res0, mismatches, redo=False):
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


def full_tune_key_finding(df=None, tunes=None, redo=False):
    """
    Evaluate the key-finding algorithm on full tunes (no train/test split).

    Builds empirical modal profiles from the cleaned corpus via
    ``KMF.get_modal_profiles``, then runs ``KMF.assign_key_and_mode`` on
    every tune's full ``tchroma`` sequence and compares the predicted
    key + mode against the ground-truth label.  All tunes are transposed
    so the tonic maps to C (pitch class 0); the ground-truth key is
    therefore ``'C'`` and the ground-truth mode is taken from the suffix
    of ``df['mode']``.

    Parameters
    ----------
    df, tunes : optional
        Cleaned setting-level metadata and matching tune feature dict as
        returned by ``load_tunes.load_thesession_data``.  If either is
        ``None``, both are loaded fresh.
    redo : bool, optional
        If ``True``, recompute even if the cache exists.  Default is
        ``False``.

    Returns
    -------
    dict
        Mapping with the evaluation results:

        * ``'y_true'``, ``'y_pred'`` — arrays of ground-truth and predicted
          mode strings (e.g. ``'Cmajor'``, ``'G#dorian'``).
        * ``'f1_joint_macro'`` — macro F1 over the four ground-truth
          joint classes (``Cmajor``/``Cmixolydian``/``Cminor``/``Cdorian``).
        * ``'f1_mode_macro'`` — macro F1 over the four mode classes,
          ignoring predicted tonic.
        * ``'acc_joint'``, ``'acc_mode'``, ``'acc_key'`` — top-1 accuracy
          for joint, mode-only, and key-only predictions.
    """
    path = PATH_FIG_DATA.joinpath("full_tune_key_finding.npz")
    if path.exists() and not redo:
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    if df is None or tunes is None:
        df, tunes = load_tunes.load_thesession_data()

    modes = list(MODES.keys())

    def mode_suffix(s):
        return s[2:] if len(s) >= 2 and s[1] == '#' else s[1:]

    def key_prefix(s):
        return s[:2] if len(s) >= 2 and s[1] == '#' else s[:1]

    df = df.copy()
    df['mode_only'] = df['mode'].apply(mode_suffix)

    # Build empirical modal profiles from the corpus; the helper applies
    # its own profile-building filters internally.
    profiles = KMF.get_modal_profiles(df, tunes)

    # Ground truth: everything is transposed so the tonic is C; the mode
    # comes from the dataset annotation.
    y_true = np.array([f"C{m}" for m in df['mode_only']])
    print(f"Evaluating key-finding on {len(df)} tunes")

    tchromas = list(df['tchroma'])
    worker = partial(KMF.assign_key_and_mode, mode_profiles=profiles)
    with Pool(N_PROC) as pool:
        y_pred = list(tqdm(pool.imap(worker, tchromas, chunksize=64),
                           total=len(tchromas)))
    y_pred = np.array(y_pred)

    # Joint F1 restricted to the four ground-truth classes so that
    # predicted-only labels (e.g. 'Gmajor' when GT is 'Cmajor') don't
    # spawn empty-support classes.
    gt_labels = [f"C{m}" for m in modes]
    f1_joint_macro = f1_score(y_true, y_pred, labels=gt_labels, average='macro', zero_division=0)
    f1_joint_weighted = f1_score(y_true, y_pred, labels=gt_labels, average='weighted', zero_division=0)
    acc_joint = float(np.mean(y_true == y_pred))

    y_true_mode = np.array([mode_suffix(s) for s in y_true])
    y_pred_mode = np.array([mode_suffix(s) for s in y_pred])
    f1_mode_macro = f1_score(y_true_mode, y_pred_mode, labels=modes, average='macro', zero_division=0)
    f1_mode_weighted = f1_score(y_true_mode, y_pred_mode, labels=modes, average='weighted', zero_division=0)
    acc_mode = float(np.mean(y_true_mode == y_pred_mode))

    y_pred_key = np.array([key_prefix(s) for s in y_pred])
    acc_key = float(np.mean(y_pred_key == 'C'))

    cm_joint = confusion_matrix(y_true, y_pred, labels=gt_labels)
    cm_mode = confusion_matrix(y_true_mode, y_pred_mode, labels=modes)

    print(f"Joint (key+mode) — acc {acc_joint:.4f}  F1 macro {f1_joint_macro:.4f}  F1 weighted {f1_joint_weighted:.4f}")
    print(f"Mode only       — acc {acc_mode:.4f}  F1 macro {f1_mode_macro:.4f}  F1 weighted {f1_mode_weighted:.4f}")
    print(f"Key only        — acc {acc_key:.4f}")
    print("Joint confusion matrix (rows=true, cols=pred):")
    print("  labels:", gt_labels)
    print(cm_joint)
    print("Mode confusion matrix (rows=true, cols=pred):")
    print("  labels:", modes)
    print(cm_mode)

    out = dict(
        y_true=y_true, y_pred=y_pred,
        gt_labels=np.array(gt_labels), mode_labels=np.array(modes),
        cm_joint=cm_joint, cm_mode=cm_mode,
        f1_joint_macro=f1_joint_macro, f1_joint_weighted=f1_joint_weighted,
        f1_mode_macro=f1_mode_macro, f1_mode_weighted=f1_mode_weighted,
        acc_joint=acc_joint, acc_mode=acc_mode, acc_key=acc_key,
    )
    np.savez(path, **out)
    return out
