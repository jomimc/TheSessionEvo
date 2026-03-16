import json
import os
import shutil
from subprocess import Popen, PIPE

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from thesession.io import tune_loader as load_tunes
from thesession.io import savage_loader as savage
from thesession.alignment import pairwise as seq_align
from thesession.io import seq_io
from thesession.analysis import substitution as SM
from thesession.config import *



########################################################################################
### Optimizing parameters for mmseqs search


def run_mmseqs(path_fasta, path_result, path_tmp, path_submat, gap_open, gap_extend):
    """
    Execute an MMseqs2 easy-search all-vs-all alignment run.

    Parameters
    ----------
    path_fasta : path-like
        Path to the FASTA file containing all encoded tune sequences.
        Used as both query and target for the all-vs-all comparison.
    path_result : path-like
        Destination file path for the MMseqs2 output (tab-separated .m8 format).
    path_tmp : path-like
        Directory for MMseqs2 temporary/index files.
    path_submat : path-like
        Path to the substitution matrix file (.out) to pass to MMseqs2.
    gap_open : int or float
        Gap-open penalty (positive number; MMseqs2 applies it as a cost).
    gap_extend : int or float
        Gap-extension penalty (positive number; must be <= ``gap_open``).

    Returns
    -------
    stdout : bytes
        Standard output captured from the MMseqs2 process.
    stderr : bytes
        Standard error captured from the MMseqs2 process.  A non-empty value
        signals that MMseqs2 encountered an error and the result file should
        not be trusted.

    Notes
    -----
    ``--format-mode 4`` requests BLAST-tab output with a header line, which
    pandas can read directly.  The binary path is taken from the project-wide
    ``MMSEQS_BIN`` config constant.
    """
    args = [MMSEQS_BIN, 'easy-search', str(path_fasta), str(path_fasta),
            str(path_result), str(path_tmp), '--format-mode', '4',
            '--sub-mat', str(path_submat), '--gap-open', str(gap_open),
            '--gap-extend', str(gap_extend)]

    pipe_output = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = pipe_output.communicate()
    return stdout, stderr


def explore_parameter_space(df, dataset='thesession_tunes'):
    """
    Grid-search substitution matrices and gap penalties using MMseqs2, caching
    per-parameter-combination results as JSON files.

    For every combination of (substitution matrix, gap_open, gap_extend) the
    function either loads a previously cached result or runs a fresh MMseqs2
    search, annotates the hits with tune-family labels, computes ROC / AUC /
    precision metrics, and appends the result dict to an in-memory list.
    Results are written to disk immediately after each run so that long grid
    searches can be interrupted and resumed without recomputing finished jobs.

    Parameters
    ----------
    df : pd.DataFrame
        Tune metadata table.  Must contain the dataset-specific identifier
        column (``tune_id`` for thesession, ``song_id`` for meertens,
        ``chapter`` for savage_english) so that within-family pair counts
        can be derived.
    dataset : str, optional
        Name of the dataset being evaluated.  Controls which subdirectory is
        used for FASTA / result files and which column is treated as the
        tune-family identifier.  One of ``'thesession_tunes'``,
        ``'meertens'``, or ``'savage_english'``.  Default is
        ``'thesession_tunes'``.

    Returns
    -------
    None
        Results are not returned; they are accumulated in the local ``data``
        list and each combination is persisted as a JSON file under
        ``PATH_CACHE / "ParameterOptimizationSearch" / dataset /``.

    Notes
    -----
    Parameter grid
        * Substitution matrices: all ``*.out`` files found in
          ``PATH_MMSEQS / "substitution_matrices"``, sorted alphabetically.
        * ``gap_open``: integers in ``[2, 15]`` inclusive.
        * ``gap_extend``: integers in ``[1, gap_open]`` inclusive, ensuring
          the extension penalty never exceeds the open penalty.

    Caching strategy
        Each combination is stored as
        ``<matrix_stem>_<gap_open>_<gap_extend>.json``.  If the file exists
        and is valid JSON it is loaded directly; if the file is corrupt
        (raises any exception during ``json.load``) it is deleted and will
        be regenerated on the next run.

    Metrics stored per combination
        * ``path_submat``, ``gap_open``, ``gap_extend`` — parameter identity.
        * ``fpr``, ``tpr`` — ROC curve arrays (lists for JSON serialisation).
        * ``auc`` — area under the ROC curve.
        * ``screened``, ``screened_positives``, ``screened_negatives`` —
          MMseqs2 hit counts split by family membership.
        * ``positives``, ``negatives``, ``total`` — exhaustive pair counts
          derived from the full ``df``.
        * ``actual_tpr``, ``actual_fpr`` — screened counts normalised by
          exhaustive counts (retrieval recall / contamination).
        * ``mean_precision`` — area under the precision-recall curve.

    Error handling
        If MMseqs2 writes anything to stderr the result dict will contain
        only the three parameter-identity keys; no metric keys are added,
        preventing downstream code from silently consuming garbage results.

    Temporary files
        MMseqs2 writes indices and intermediate results to
        ``path_mmseqs.with_name("tmp")``.  This directory is removed with
        ``shutil.rmtree`` after each successful (or failed) run so that stale
        index files from one matrix cannot contaminate the next run.
    """
    # Set up paths to input / outputs
    path_base = PATH_MMSEQS.joinpath(f'{dataset}')
    path_fasta = path_base.joinpath(f"all_seq_{dataset}.fasta")
    # MMseqs2 writes its raw hit table here; overwritten on every run
    path_mmseqs = PATH_MMSEQS.joinpath("tmp", "result.m8")
    path_mmseqs.parent.mkdir(parents=True, exist_ok=True)

    # Collect all candidate substitution matrices; sort for reproducible ordering
    path_submat_list = sorted(PATH_MMSEQS.joinpath("substitution_matrices").glob("*.out"))
    path_results = PATH_CACHE.joinpath("ParameterOptimizationSearch", dataset)
    path_results.mkdir(parents=True, exist_ok=True)

    # Load tune family annotations
    families, family_key = seq_io.get_families_key(df, dataset)
    # Map dataset name to the column that identifies individual tunes (for pair counting)
    xkey = {'thesession_tunes':'tune_id',
            'meertens':'song_id',
            'savage_english':'chapter'
            }[dataset]

    gap_open = np.arange(2, 16, 1)
    data = []
    for path_submat in path_submat_list:
        for go in gap_open:
            # gap_extend must be <= gap_open; arange upper-bound is exclusive so +1
            gap_extend = np.arange(1, go + 1, 1)
            for ge in gap_extend:
                # Unique cache file for this (matrix, go, ge) triple
                path = path_results.joinpath(f"{path_submat.stem}_{go}_{ge}.json")
                print(path)
                if path.exists():
                    try:
                        data.append(json.load(open(path, 'r')))
                    except:
                        # Corrupt JSON (e.g. partial write from a previous crash); delete and rerun
                        os.remove(path)

                else:
                    print(path)
                    stdout, stderr = run_mmseqs(path_fasta, path_mmseqs, path_mmseqs.with_name("tmp"),
                                                path_submat, go, ge)
                    print(stdout)
                    print(stderr)

                    # Initialise result dict with parameter identity so the cache file
                    # is always written, even when MMseqs2 fails
                    out = {'path_submat':str(path_submat), 'gap_open':float(go), 'gap_extend':float(ge)}

                    if len(stderr) == 0:
                        # Parse the MMseqs2 hit table and label each pair by family membership
                        res = pd.read_csv(path_mmseqs, sep='\t')
                        # Remove self-hits
                        res = res.loc[res['query'] != res['target']]
                        # Remove reverse duplicates: keep only one of (A, B) and (B, A)
                        pairs = res[['query', 'target']].values
                        df_sort = pd.DataFrame(np.sort(pairs, axis=1), columns=['a', 'b'])
                        res = res.loc[df_sort.duplicated().values == False]
                        res = seq_io.annotate_alignment(res, families, family_key)

                        # Compute ROC curve and screened-pair counts using fident as score
                        out = get_roc_and_auc(res, dataset, out)

                        # Add exhaustive positive/negative pair counts derived from df
                        out = get_total_positives(df, dataset, xkey, out)

                        # Overall AUC with fident as the discrimination score
                        out['auc'] = float(roc_auc_score(res.in_fam, res.fident))

                        # Actual TPR/FPR: what fraction of all possible pairs did MMseqs2 return?
                        tpr, fpr = calculate_actual_rates(out)
                        out['actual_tpr'] = tpr
                        out['actual_fpr'] = fpr

                        # Area under the precision-recall curve
                        out['mean_precision'] = calculate_average_precision(out)

                    data.append(out)
                    json.dump(out, open(path, 'w'))
                    # Remove MMseqs2 temp directory so stale indices don't bleed into the next run
                    shutil.rmtree(path_mmseqs.with_name("tmp"))


def get_total_positives(df, dataset, x='tune_id', fig_data={}):
    """
    Count exhaustive positive and negative pairs implied by the tune-family labels.

    Parameters
    ----------
    df : pd.DataFrame
        Tune metadata table with a column ``x`` that identifies each tune's
        family (or individual tune ID used as a family proxy).
    dataset : str
        Dataset name; currently unused in the computation but reserved for
        future dataset-specific logic.
    x : str, optional
        Column name in ``df`` whose repeated values define tune families.
        Default is ``'tune_id'``.
    fig_data : dict, optional
        Existing result dictionary to update in-place.  Default is an empty dict.

    Returns
    -------
    fig_data : dict
        Updated dictionary with three new integer keys:
        ``'positives'`` — number of within-family pairs,
        ``'negatives'`` — number of cross-family pairs,
        ``'total'``     — total number of ordered pairs (``len(df) ** 2``).

    Notes
    -----
    Positives are counted as the number of *unordered* within-family pairs:
    for a family of size *n* this is n*(n-1)/2.  The total, however, is the
    number of *ordered* pairs (``N^2``) including self-comparisons, matching
    the all-vs-all search space that MMseqs2 explores.  Negatives are
    defined as the complement: ``total - positives``.
    """
    N = len(df)
    total = N * (N - 1) / 2
    # Sum n*(n-1)/2 over every family to get the total within-family (positive) pair count
    positives = np.sum([n * (n - 1) / 2 for n in df[x].value_counts().values])
    negatives = total - positives
    fig_data[f'positives'] = int(positives)
    fig_data[f'negatives'] = int(negatives)
    fig_data[f'total'] = int(total)
    return fig_data


def get_roc_and_auc(res, dataset, fig_data={}):
    """
    Compute ROC curve and AUC from alignment results, and record screened-pair counts.

    Parameters
    ----------
    res : pd.DataFrame
        MMseqs2 hit table annotated with an ``in_fam`` boolean column (True if
        the query and target belong to the same tune family) and an ``fident``
        column (fractional sequence identity used as the alignment score).
    dataset : str
        Dataset name; currently unused in the computation but reserved for
        future dataset-specific logic.
    fig_data : dict, optional
        Existing result dictionary to update in-place.  Default is an empty dict.

    Returns
    -------
    fig_data : dict
        Updated dictionary with the following keys added:
        ``'fpr'`` — list of false positive rates along the ROC curve,
        ``'tpr'`` — list of true positive rates along the ROC curve,
        ``'auc'`` — area under the ROC curve (float),
        ``'screened'`` — total number of pairs returned by MMseqs2,
        ``'screened_positives'`` — within-family pairs returned by MMseqs2,
        ``'screened_negatives'`` — cross-family pairs returned by MMseqs2.

    Notes
    -----
    ``fident`` (fractional sequence identity) is used as the classifier score
    throughout; higher values indicate stronger alignment and are expected to
    correlate with within-family membership.  ROC arrays are converted to
    plain Python lists so the dict can be serialised directly to JSON.
    """
    fpr, tpr, _ = roc_curve(res.in_fam, res.fident)
    auc = roc_auc_score(res.in_fam, res.fident)

    # Save to container
    fig_data[f'fpr'] = list(fpr)
    fig_data[f'tpr'] = list(tpr)
    fig_data[f'auc'] = auc
    fig_data[f'screened'] = len(res)
    fig_data[f'screened_positives'] = int(np.sum(res.in_fam))
    # Screened negatives are the complement of screened positives within returned hits
    fig_data[f'screened_negatives'] = int(len(res) - np.sum(res.in_fam))
    return fig_data


def calculate_actual_rates(data):
    """
    Compute the true positive rate and false positive rate from screened alignment results.

    Parameters
    ----------
    data : dict
        Results dictionary containing ``'screened_positives'``,
        ``'screened_negatives'``, ``'positives'``, and ``'negatives'`` keys
        (no dataset prefix, unlike the evaluation.py counterpart).

    Returns
    -------
    tpr : float
        True positive rate: screened positives / total positives.
    fpr : float
        False positive rate: screened negatives / total negatives.

    Notes
    -----
    These rates quantify MMseqs2's retrieval behaviour before any score
    threshold is applied: TPR measures how many true within-family pairs were
    retrieved; FPR measures how many cross-family pairs were inadvertently
    retrieved.
    """
    # Divide reported pairs by the exhaustive pair counts to get rates
    tpr = data[f"screened_positives"] / data[f"positives"]
    fpr = data[f"screened_negatives"] / data[f"negatives"]
    return tpr, fpr


def get_precision_recall(data):
    """
    Compute precision and recall curves from ROC data and pair counts.

    Parameters
    ----------
    data : dict
        Results dictionary containing integer pair counts (``'total'``,
        ``'positives'``, ``'negatives'``, ``'screened_positives'``,
        ``'screened_negatives'``) and ROC arrays (``'fpr'``, ``'tpr'``).
        No dataset prefix is used (unlike the evaluation.py counterpart).

    Returns
    -------
    precision : np.ndarray
        Precision at each ROC operating point: TP / (TP + FP).
    recall : np.ndarray
        Recall (= TPR) at each ROC operating point: TP / total positives.

    Notes
    -----
    Precision and recall are derived by combining the ROC curve with the
    absolute screened pair counts (``Mt`` positives, ``Mf`` negatives).
    ``N`` (total pairs) is unpacked for completeness but is not used in the
    calculation.  ROC arrays are read from the dict as lists and converted
    to numpy arrays here to support vectorised arithmetic.
    """
    names = ['total', 'positives', 'negatives', 'screened_positives',
             'screened_negatives']
    N, Nt, Nf, Mt, Mf = [data[x] for x in names]
    # ROC arrays are stored as plain lists for JSON compatibility; convert back to arrays
    fpr, tpr = [np.array(data[x]) for x in ['fpr', 'tpr']]

    # Convert ROC rates back to absolute TP/FP counts using the screened set
    # sizes, then form precision and recall at each threshold point
    denom = tpr * Mt + fpr * Mf
    precision = np.divide(tpr * Mt, denom, out=np.zeros_like(tpr), where=denom > 0)
    recall = (tpr * Mt) / Nt
    return precision, recall


def calculate_average_precision(data):
    """
    Compute the area under the precision-recall curve via the step method.

    Parameters
    ----------
    data : dict
        Results dictionary passed through to ``get_precision_recall``.

    Returns
    -------
    ap : float
        Average precision: the weighted mean of precisions at each recall
        threshold, approximated as the sum of rectangular steps.

    Notes
    -----
    Uses a right-hand Riemann sum (``precision[1:]`` paired with
    ``recall[1:] - recall[:-1]``), which matches scikit-learn's
    ``average_precision_score`` convention.
    """
    precision, recall = get_precision_recall(data)
    # Sum rectangular strips under the precision-recall curve (right-hand rule)
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])


def parse_filename(filename):
    """
    Split a result filename stem into its constituent parameter components.

    Parameters
    ----------
    filename : str
        Stem of a cached result filename, expected to follow the convention
        ``<mat_type>_<diag>_<off_diag>_<gap_open>_<gap_extend>``
        (e.g. ``"A_4_-1_8_2"``).

    Returns
    -------
    mat_type : str
        Substitution matrix family identifier (e.g. ``'A'`` or ``'B'``).
    diag : str
        Diagonal (match) score encoded in the filename.
    off_diag : str
        Off-diagonal (mismatch) score encoded in the filename.
    gap_open : str
        Gap-open penalty encoded in the filename.
    gap_extend : str
        Gap-extension penalty encoded in the filename.

    Notes
    -----
    All returned values are strings; callers are responsible for casting to
    numeric types if needed.  The function assumes exactly five
    underscore-delimited fields; malformed filenames will raise a
    ``ValueError`` from the unpacking.
    """
    mat_type, diag, off_diag, gap_open, gap_extend = filename.split('_')
    return mat_type, diag, off_diag, gap_open, gap_extend


def load_results_mmseqs(dataset):
    """
    Load all cached MMseqs2 parameter-search results for a dataset into a DataFrame.

    Parameters
    ----------
    dataset : str
        Dataset name used to locate the cache directory:
        ``PATH_CACHE / "ParameterOptimizationSearch" / dataset``.

    Returns
    -------
    pd.DataFrame
        One row per (matrix, gap_open, gap_extend) combination.  Each row
        contains all keys stored in the JSON result file plus three additional
        columns parsed from the filename: ``'mat_type'``, ``'diag'``, and
        ``'off_diag'``.

    Notes
    -----
    Files are processed in sorted order (matching the order used during
    ``explore_parameter_space``) to give a deterministic row ordering.
    The ``gap_open`` and ``gap_extend`` fields remain as strings from
    ``parse_filename``; the numeric values are also present in the JSON
    payload as floats under the same keys.
    """
    path_results = PATH_CACHE.joinpath("ParameterOptimizationSearch", dataset)
    path_list = sorted(path_results.glob("*json"))
    data = []
    for path in path_list:
        # Recover matrix metadata from the filename rather than re-parsing the JSON payload
        mat_type, diag, off_diag, gap_open, gap_extend = parse_filename(path.stem)
        out = json.load(open(path, 'r'))
        out['mat_type'] = mat_type
        out['diag'] = diag
        out['off_diag'] = off_diag
        data.append(out)
    return pd.DataFrame(data)


########################################################################################
### Optimizing parameters for NW pairwise alignments


def optimize_alignment_savage():
    """
    Grid-search Needleman-Wunsch alignment parameters against the Savage English dataset.

    For every (match, mismatch, gap_open, gap_extend) combination produced by
    ``params_savage``, runs a global pairwise alignment of all tune pairs in the
    Savage dataset and saves the resulting correctness and alignment-count arrays
    as a ``.npy`` file under ``PATH_CACHE / "ParameterOptimizationAlignment"``.

    Returns
    -------
    None
        Results are written to disk as ``.npy`` files; nothing is returned.

    Notes
    -----
    Each saved array has shape ``(2, n_pairs)`` where row 0 is a binary
    ``correct`` indicator and row 1 is ``num_align`` (number of optimal
    alignments).  The integer cast ensures compact storage.
    The Savage dataset contains manually curated aligned folk-song variants,
    making it a gold standard for evaluating whether alignment parameters
    recover the expected groupings.
    """
    df = savage.load_df_aligned()
    path_results = PATH_CACHE.joinpath("ParameterOptimizationAlignment")
    params = params_savage()

    for ma, mi, go, ge in params:
        path = path_results.joinpath(f"{ma}_{mi}_{go}_{ge}.npy")
        print(path)
        kwargs = {'match':ma, 'mismatch':mi, 'gap_open':go,
                  'gap_extend':ge, 'alg':'global'}
        np.save(path, np.array(savage.compare_all_alignments(df, **kwargs), int))


def params_savage():
    """
    Generate the full grid of Needleman-Wunsch parameter combinations for the Savage search.

    Returns
    -------
    params : list of tuple
        List of ``(match, mismatch, gap_open, gap_extend)`` tuples covering
        all valid combinations in the search grid.  ``gap_extend`` is always
        <= ``gap_open``.

    Notes
    -----
    Parameter ranges:

    * ``match``      : even integers in ``[2, 10]``
    * ``mismatch``   : integers in ``[-4, 0]``
    * ``gap_open``   : integers in ``[2, 15]``
    * ``gap_extend`` : integers in ``[1, gap_open]`` (inclusive)

    The constraint ``gap_extend <= gap_open`` enforces the standard affine-gap
    convention where opening a gap is at least as costly as extending one.
    """
    match = np.arange(2, 12, 2)
    mismatch = np.arange(-4, 1, 1)
    gap_open = np.arange(2, 16, 1)
    params = []
    for ma in match:
        for mi in mismatch:
            for go in gap_open:
                # gap_extend upper-bound is go+1 so that go itself is included
                gap_extend = np.arange(1, go + 1, 1)
                for ge in gap_extend:
                    params.append((ma, mi, go, ge))
    return params


def load_results_savage():
    """
    Load all cached Needleman-Wunsch parameter-search results for the Savage dataset.

    Returns
    -------
    dfr : pd.DataFrame
        One row per parameter combination with columns for each parameter
        (``match``, ``mismatch``, ``gap_open``, ``gap_extend``), the cache
        file path (``path``), mean fraction correct (``frac_correct``), and
        mean number of optimal alignments (``mean_num_align``).
    freq_correct : np.ndarray
        Per-tune-pair mean correctness across all parameter combinations,
        indicating which pairs are consistently easy or hard to align correctly.

    Notes
    -----
    ``correct`` and ``num_align`` are loaded from ``.npy`` files written by
    ``optimize_alignment_savage``.  ``freq_correct`` is computed by averaging
    the ``correct`` arrays along axis 0 (across parameter combinations), so its
    length equals the number of tune pairs in the Savage dataset.
    """
    path_results = PATH_CACHE.joinpath("ParameterOptimizationAlignment")
    params = params_savage()
    output = []
    overall = []
    for ma, mi, go, ge in params:
        path = path_results.joinpath(f"{ma}_{mi}_{go}_{ge}.npy")
        correct, num_align = np.load(path)
        kwargs = {'match':ma, 'mismatch':mi, 'gap_open':go,
                  'gap_extend':ge}
        kwargs.update({'path':path, 'frac_correct':np.mean(correct), 'mean_num_align':np.mean(num_align)})
        output.append(kwargs)
        # Accumulate per-pair correctness arrays to later compute cross-parameter averages
        overall.append(correct)
    dfr = pd.DataFrame(data=output)
    # Average correctness per tune pair across all parameter combinations
    freq_correct = np.mean(overall, axis=0)
    return dfr, freq_correct


if __name__ == "__main__":

    if 0:
        # Generate substitution matrices for evaluation
        SM.generate_all_sub_mat()

    # TheSession
    if 0:
        # Load thesession data
        df, data = load_tunes.load_thesession_data()

        # Run mmseqs with different substitution matrices and evaluate results
        explore_parameter_space(df)


    # Meertens
    if 1:
        # Load thesession data
        df, data = load_tunes.load_meertens_data()

        # Run mmseqs with different substitution matrices and evaluate results
        explore_parameter_space(df, 'meertens')


    # Bronson
    if 1:
        # Load thesession data
        df = load_tunes.load_bronson_data()

        # Run mmseqs with different substitution matrices and evaluate results
        explore_parameter_space(df, 'bronson')

    # Optimize alignment parameters for replicating Pat's manual alignments
    if 0:
        optimize_alignment_savage()
