from collections import Counter, defaultdict
from itertools import product
import json
from math import comb
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.alignment import pairwise as seq_align
from thesession.io import seq_io
from thesession import utils



######################################################################
### Generating simple substitution matrices

### Fixed match and mismatch scores for all substitutions
def basic_submat_A(diag=6, off_diag=-1.5, nmax=12, noise=0):
    """
    Build a flat substitution matrix where every match scores the same and
    every mismatch scores the same, regardless of the interval distance
    between notes.

    Parameters
    ----------
    diag : float, optional
        Score assigned to identical-note substitutions (the diagonal).
        Default is 6.
    off_diag : float, optional
        Score assigned to all non-identical substitutions. Default is -1.5.
    nmax : int, optional
        Number of pitch-class positions (chromatic notes, 0-11). Positions
        beyond nmax correspond to non-pitch alphabet letters and are left at
        0. Default is 12.
    noise : float, optional
        Half-amplitude of uniform random noise added to every cell, useful
        for breaking symmetry during optimisation experiments. When 0 (the
        default) no noise is added.

    Returns
    -------
    basic : numpy.ndarray, shape (N, N)
        Substitution score matrix, where N = len(letters) (the full
        protein-alphabet proxy used by MMseqs2).

    Notes
    -----
    The matrix spans the entire protein-alphabet letter set imported from
    ``config``, not just the 12 chromatic pitch classes, so that it can be
    written directly to an MMseqs2-compatible file.
    """
    N = len(letters)
    # Initialise the entire N×N matrix to the off-diagonal penalty, then
    # overwrite the diagonal with the match reward.
    basic = np.zeros((N, N)) + off_diag
    np.fill_diagonal(basic, diag)
    if noise > 0:
        # Add a teeny bit of noise
        basic = basic + (np.random.rand(basic.size) - 0.5).reshape(basic.shape) * noise
    return basic


### Mismatch scores depend on substitution distance (semitones)
def basic_submat_B(diag=6, off_diag=-0.5, nmax=12, noise=0):
    """
    Build a distance-weighted substitution matrix where the mismatch penalty
    scales with the semitone interval between two pitch classes.

    The penalty for substituting note i with note j is
    ``min(|i-j|, 12-|i-j|) * off_diag``, i.e. the shorter of the two
    chromatic paths around the octave multiplied by the per-semitone cost.

    Parameters
    ----------
    diag : float, optional
        Score assigned to identical-note substitutions. Default is 6.
    off_diag : float, optional
        Per-semitone penalty multiplier (should be negative or zero so that
        larger intervals receive a larger penalty). Default is -0.5.
    nmax : int, optional
        Number of pitch-class positions treated as chromatic notes. Cells
        outside this range are left at 0. Default is 12.
    noise : float, optional
        Half-amplitude of uniform random noise added to every cell.
        Default is 0 (no noise).

    Returns
    -------
    basic : numpy.ndarray, shape (N, N)
        Distance-weighted substitution score matrix.

    Notes
    -----
    Using ``min(d, 12-d)`` ensures that interval distances are measured as
    the shortest path on the pitch-class circle (e.g., a tritone is 6
    semitones in either direction, a major seventh is treated as a minor
    second in the opposite direction, distance = 1).
    """
    N = len(letters)
    basic = np.zeros((N, N))
    for i, j in product(range(N), range(N)):
        if (i < nmax) and (j < nmax):
            d = abs(i - j)
            # Use the shorter arc around the chromatic circle so that, for
            # example, B->C (11 semitones up, but 1 semitone down) scores
            # the same as C->C# (1 semitone up).
            basic[i,j] = min(d, abs(12 - d)) * off_diag
    np.fill_diagonal(basic, diag)
    if noise > 0:
        # Add a teeny bit of noise
        basic = basic + (np.random.rand(basic.size) - 0.5).reshape(basic.shape) * noise
    return basic


### Write a substitution matrix in the correct format for mmseqs
def write_mmseqs_sub_mat(path, submat, nmax=12):
    """
    Serialise a substitution matrix to an MMseqs2-compatible text file.

    MMseqs2 expects a specific plain-text header format that includes an
    optional pre-computed background frequency vector and a lambda value,
    followed by a labelled matrix body. This function writes that exact
    layout so the file can be passed directly to ``mmseqs search`` via
    ``--sub-mat``.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file path.
    submat : numpy.ndarray, shape (N, N)
        Score matrix whose rows/columns correspond to ``letters`` from
        ``config`` (protein-alphabet proxy encoding for pitch classes).
    nmax : int, optional
        Number of pitch classes with non-zero background frequency (i.e.
        the 12 chromatic tones). All positions beyond nmax receive a
        background frequency of 0. Default is 12.

    Returns
    -------
    None

    Notes
    -----
    The background frequency is set to a uniform 1/nmax for each of the
    first ``nmax`` positions, reflecting the assumption that all 12
    pitch classes are equally likely a priori (before any corpus-derived
    frequencies are available).

    The lambda value (0.34657 ≈ ln 2 / 2) is a standard Karlin-Altschul
    parameter; it is written as a comment so MMseqs2 may use it for
    E-value calculation if desired.
    """
    # Build a uniform background over the nmax pitch-class positions;
    # non-pitch letters get zero background weight.
    background = np.zeros(len(letters))
    background[:nmax] = 1 / nmax
    background_txt = ' '.join(["# Background (precomputed optional):"] + [f"{p:7.5f}" for p in background])
    with open(path, 'w') as o:
        o.write("# Custom substitution matrix\n")
        o.write(background_txt + '\n')
        o.write("# Lambda     (precomputed optional): 0.34657\n")
        # Write the column-header row; rstrip removes trailing whitespace.
        o.write(('  ' + ' '.join(f" {l}     " for l in letters)).rstrip())
        for row, l in zip(submat, letters):
            o.write('\n')
            o.write(' '.join([f"{l}"] + [f"{item:7.4f}" for item in row]))


### Generate many substitution matrices for optimization
def generate_all_sub_mat():
    """
    Generate and save a grid of type-A and type-B substitution matrices for
    parameter-space optimisation.

    Type-A matrices use a constant off-diagonal penalty; type-B matrices
    use a distance-weighted penalty (see ``basic_submat_A`` and
    ``basic_submat_B``). Files are written to
    ``PATH_MMSEQS/substitution_matrices/`` using the naming convention
    ``A_<diag>_<off_diag>.out`` and ``B_<diag>_<off_diag>.out``.

    Returns
    -------
    None

    Notes
    -----
    The parameter grids are hard-coded to cover a range of biologically
    plausible score combinations:

    - Type A: diagonal in {2, 4, 6, 8, 10}, off-diagonal in {-4, -3, -2, -1, 0}
    - Type B: diagonal in {1, 2, 3, 4, 5}, per-semitone cost in [-1, 0) step 0.2
    """
    path_base = PATH_MMSEQS.joinpath("substitution_matrices")
    path_base.mkdir(parents=True, exist_ok=True)
    diag_arr = np.arange(2, 12, 2)

    ### First, generate type A, where off-diagonal is constant
    off_diag = np.arange(-4, 1, 1)
    for d in diag_arr:
        for od in off_diag:
            path = path_base.joinpath(f"A_{d}_{od}.out")
            submat = basic_submat_A(d, od)
            write_mmseqs_sub_mat(path, submat)

    ### Second, generate type B, where mismatch is distance dependent
    diag_arr = np.arange(1, 6, 1)
    off_diag = np.arange(-1, 0, 0.2)
    for d in diag_arr:
        for od in off_diag:
            path = path_base.joinpath(f"B_{d}_{od:4.1f}.out")
            submat = basic_submat_B(d, od)
            write_mmseqs_sub_mat(path, submat)



######################################################################
### Get substitution rates / matrices from MSA


### Count substitution rates from MSA
def count_substitutions_from_msa(msa, gap_max=0.3, observations=None):
    """
    Accumulate pairwise note-substitution counts from a multiple sequence
    alignment (MSA) using the star-alignment counting method.

    For every alignment column that passes the gap-frequency filter, every
    unordered pair of non-gap residues observed in that column is counted as
    one substitution event. Same-note pairs (i.e. conserved positions)
    contribute to the diagonal of the resulting substitution count
    dictionary and are included so that the diagonal can later be used to
    estimate note frequencies.

    Parameters
    ----------
    msa : numpy.ndarray, shape (n_seqs, n_cols)
        Multiple sequence alignment. Elements are either byte-string
        characters (``dtype == np.string_``, e.g. b'A', b'-') or
        floating-point pitch-class integers with ``np.nan`` representing
        gaps.
    gap_max : float, optional
        Maximum fraction of gaps allowed in a column before that column is
        discarded. Default is 0.3 (columns with more than 30 % gaps are
        ignored).
    observations : collections.defaultdict or None, optional
        Pre-existing observation dictionary to accumulate into. Passing an
        existing dict allows results from multiple alignments or MSA
        segments to be merged. If None, a fresh ``defaultdict(int)`` is
        created. Default is None.

    Returns
    -------
    observations : collections.defaultdict
        Dictionary mapping ``(note_a, note_b)`` pairs — with
        ``note_a <= note_b`` to avoid double-counting — to integer counts.
        Diagonal entries ``(note, note)`` count same-note co-occurrences;
        off-diagonal entries count heterologous substitutions.

    Notes
    -----
    The counting strategy mirrors the approach used in protein substitution
    matrix construction (e.g. BLOSUM): for each alignment column we treat
    every pair of sequences as one independent observation. Formally, if
    note k appears c_k times in a column, the number of same-note pairs is
    C(c_k, 2) = c_k*(c_k-1)/2 (binomial coefficient), and the number of
    cross-note pairs between notes k1 and k2 is c_k1 * c_k2.

    Ordering keys as ``(min, max)`` ensures that the dictionary encodes
    only the upper triangle of the substitution matrix, avoiding
    double-counting when the matrix is later symmetrised.

    The MSA may use either the protein-alphabet letter proxy (byte strings)
    or raw pitch-class integers (floats with NaN gaps); the appropriate
    gap-detection function is selected automatically from ``msa.dtype``.
    """
    # Choose gap-detection strategy based on array dtype:
    # byte-string MSAs use '-' as the gap character, whereas float MSAs
    # use NaN so that arithmetic operations still work.
    if msa.dtype == np.string_:
        gap = np.mean(msa == '-', axis=0)
        gap_fn = lambda x: x == '-'
    else:
        gap = np.mean(np.isnan(msa), axis=0)
        gap_fn = lambda x: np.isnan(x)

    # Drop columns whose gap fraction exceeds the threshold; highly gapped
    # columns contribute noise rather than signal to substitution counts.
    msa = msa[:,gap < gap_max]
    if isinstance(observations, type(None)):
        observations = defaultdict(int)

    # Iterate over columns (transposed so each 'row' is one alignment column).
    for row in msa.T:
        count = Counter(row)
        # Exclude gap tokens from counting — gaps are not substitution events.
        keys = [k for k in count.keys() if not gap_fn(k)]

        # Add all of the same-note 'substitutions'
        for k in keys:
            # C(n, 2) gives the number of distinct sequence pairs that both
            # carry note k at this position — each such pair is one
            # observation of a same-note (diagonal) event.
            observations[(k,k)] += comb(count[k], 2)

        # Add the different-note 'substitutions'
        for i, k1 in enumerate(keys[:-1]):
            for k2 in keys[i+1:]:
                # Every sequence carrying k1 is paired with every sequence
                # carrying k2; store the canonical (min, max) key so each
                # pair is counted only once.
                observations[(min(k1, k2), max(k1, k2))] += count[k1] * count[k2]
    return observations


### Convert substitution observations (dictionary of counts)
### to a substitution matrix
def convert_observations_to_matrix(observations, chroma=False):
    """
    Convert a substitution-count dictionary into a symmetric 2-D numpy
    matrix.

    Parameters
    ----------
    observations : dict
        Dictionary mapping ``(note_a, note_b)`` pairs to integer counts, as
        produced by ``count_substitutions_from_msa`` or
        ``count_subs_pairwise_str``. Gap tokens (``'-'``) are ignored.
    chroma : bool, optional
        If True, fold all notes onto the 12 pitch classes (0–11) by taking
        ``note % 12``. This collapses octave-equivalent notes and produces a
        12×12 matrix indexed by chroma integer. If False (the default), the
        matrix is indexed by the sorted set of unique symbols found in
        ``observations``.

    Returns
    -------
    letters : numpy.ndarray or list
        Ordered sequence of row/column labels. Integer array 0–11 when
        ``chroma=True``; sorted list of symbols otherwise.
    mat : numpy.ndarray, shape (N, N)
        Symmetric substitution count matrix. ``mat[i, j]`` holds the total
        number of observed (i, j) substitution events.

    Notes
    -----
    Because ``observations`` stores only the upper-triangle key
    ``(min, max)``, writing to both ``mat[i,j]`` and ``mat[j,i]`` restores
    full symmetry. When ``chroma=True``, multiple input keys may map to the
    same chroma pair, so counts are accumulated with ``+=`` rather than
    assigned directly.
    """
    if chroma:
        letters = np.arange(12)
    else:
        # Collect every symbol that appears in any key of the observations dict.
        letters = sorted(set([x for y in observations.keys() for x in y]))
    key = {l:i for i, l in enumerate(letters)}
    mat = np.zeros((len(key), len(key)), float)
    for (k1, k2), v in observations.items():
        if k1 == '-' or k2 == '-':
            continue
        if chroma:
            # Map to pitch class 0-11, accumulating across octaves if needed.
            i, j = [int(k%12) for k in [k1, k2]]
            mat[i,j] += v
            mat[j,i] += v
        else:
            i, j = key[k1], key[k2]
            mat[i,j] = v
            mat[j,i] = v
    return letters, mat


######################################################################
### Get substitution rates / matrices from pairwise alignments

def count_subs_pairwise_str(al1, al2, obs=None):
    """
    Accumulate substitution counts from a single pairwise alignment given as
    two aligned strings.

    Parameters
    ----------
    al1 : str or iterable of str
        First aligned sequence, including gap characters (``'-'``).
    al2 : str or iterable of str
        Second aligned sequence, same length as ``al1``.
    obs : collections.defaultdict or None, optional
        Existing observation dictionary to accumulate into. If None, a fresh
        ``defaultdict(int)`` is created. Default is None.

    Returns
    -------
    obs : collections.defaultdict
        Updated dictionary mapping ``(note_a, note_b)`` — with
        ``note_a <= note_b`` — to integer counts. Each aligned residue pair
        (excluding positions where either sequence has a gap) contributes 1
        to the count.

    Notes
    -----
    Gap columns are skipped entirely: only aligned residue–residue positions
    contribute to the substitution tally, consistent with the convention
    used in ``count_substitutions_from_msa``.
    """
    if isinstance(obs, type(None)):
        obs = defaultdict(int)
    for a, b in zip(al1, al2):
        if (a != '-') and (b != '-'):
            # Store canonical (min, max) key to keep the dictionary
            # upper-triangular and avoid double-counting.
            obs[(min(a, b), max(a, b))] += 1
    return obs


######################################################################
### Get substitution rates as a function of substitution distance


### Get substituion rate as a function of substitution distance (semitones)
### MSA should have be in sequence letter format (A, B, etc.)
def get_substition_distance_letter(msa, gap_max=0.3, obs=None):
    """
    Compute the normalised substitution-rate spectrum as a function of
    semitone distance from an MSA encoded with protein-alphabet proxy
    letters.

    Parameters
    ----------
    msa : numpy.ndarray
        Multiple sequence alignment whose elements are protein-alphabet proxy
        letters (e.g. ``'A'`` = C, ``'B'`` = C#, …) as used by MMseqs2.
        Ignored when ``obs`` is provided.
    gap_max : float, optional
        Maximum gap fraction per column passed to
        ``count_substitutions_from_msa``. Default is 0.3.
    obs : dict or None, optional
        Pre-computed observation dictionary (as returned by
        ``count_substitutions_from_msa``). When supplied, ``msa`` is not
        used and the conversion/counting steps are skipped. Default is None.

    Returns
    -------
    X : numpy.ndarray, shape (6,)
        Semitone distances 1 through 6 (the maximum unambiguous interval on
        the pitch-class circle).
    Y : numpy.ndarray, shape (6,)
        Fraction of all heterologous substitutions occurring at each
        distance, summing to 1.

    Notes
    -----
    Distances are folded onto [1, 6] via ``min(d, 12-d)`` so that each
    interval is represented by its shortest chromatic path (e.g. a major
    seventh maps to distance 1). The ``position_key`` lookup (from
    ``config``) translates proxy letters back to chromatic semitone
    positions before the distance is computed.
    """
    if isinstance(obs, type(None)):
        msa = convert_msa_letters(msa)
        obs = count_substitutions_from_msa(msa, gap_max)

    dist_sub = defaultdict(int)
    for k, v in obs.items():
        # Skip gap-containing keys and diagonal (same-note) entries.
        if ('-' in k) or (k[0] == k[1]):
            continue
        dist = abs(position_key[k[0]] - position_key[k[1]])
        # Fold onto the shorter arc of the pitch-class circle.
        dist = min(dist, abs(12 - dist))
        dist_sub[dist] += v

    X = np.arange(1, 7)
    Y = np.array([dist_sub.get(x, 0) for x in X])
    # Normalise to a probability distribution over the six distance bins.
    Y = Y / Y.sum()
    return X, Y


### Get substituion rate as a function of substitution distance (semitones)
### MSA should have float dtype
def get_substition_distance(msa, gap_max=0.3, obs=None):
    """
    Compute the normalised substitution-rate spectrum as a function of
    semitone distance from an MSA encoded as floating-point pitch-class
    integers.

    Parameters
    ----------
    msa : numpy.ndarray
        Multiple sequence alignment with float dtype where each element is a
        pitch-class integer (0.0–11.0) and ``np.nan`` represents a gap.
        Ignored when ``obs`` is provided.
    gap_max : float, optional
        Maximum gap fraction per column passed to
        ``count_substitutions_from_msa``. Default is 0.3.
    obs : dict or None, optional
        Pre-computed observation dictionary. When supplied, ``msa`` is not
        used. Default is None.

    Returns
    -------
    X : numpy.ndarray, shape (17,)
        Semitone distances 1 through 17.
    Y : numpy.ndarray, shape (17,)
        Fraction of all heterologous substitutions at each distance,
        summing to 1.

    Notes
    -----
    Unlike ``get_substition_distance_letter``, distances here are *not*
    folded onto the shorter chromatic arc; the raw absolute difference
    between integer pitch-class values is used. The range [1, 17] is fixed
    rather than data-driven (the commented-out data-driven alternative is
    retained for reference) to ensure consistent array lengths across
    datasets.
    """
    if isinstance(obs, type(None)):
        obs = count_substitutions_from_msa(msa, gap_max)

    dist_sub = defaultdict(int)
    for k, v in obs.items():
        # Skip gap tokens and same-note (diagonal) pairs.
        if ('-' in k) or (k[0] == k[1]):
            continue
        elif np.any(np.isnan(k)):
            continue
        dist = abs(int(k[0]) - int(k[1]))
        dist_sub[dist] += v


#   xmin, xmax = [fn(dist_sub.keys()) for fn in [min, max]]
    xmin, xmax = [1, 17]
    X = np.arange(xmin, xmax + 1, 1)
    Y = np.array([dist_sub.get(x, 0) for x in X])
    # Normalise to a probability distribution over the distance bins.
    Y = Y / Y.sum()
    return X, Y


######################################################################
### Calculate per-note mutability and frequency from a substitution matrix

### Count the diagonal elements (same-note pairs) and off-diagonals (substitutions)
def calculate_mutability_and_frequency(mat):
    """
    Derive per-note mutability and marginal frequency from a raw
    substitution count matrix.

    Parameters
    ----------
    mat : numpy.ndarray, shape (N, N)
        Symmetric substitution count matrix as produced by
        ``convert_observations_to_matrix``. Diagonal entries hold
        same-note pair counts; off-diagonal entries hold cross-note pair
        counts.

    Returns
    -------
    mutability : numpy.ndarray, shape (N,)
        Fraction of observations for each note that involve a change, i.e.
        ``off_diagonal_count / total_count``. A value near 0 indicates a
        highly conserved note; a value near 1 indicates a highly mutable
        one.
    frequency : numpy.ndarray, shape (N,)
        Estimated marginal frequency (proportional to occurrence count) for
        each note. Diagonal counts are weighted by 2 because a same-note
        pair represents two sequence positions carrying that note.

    Notes
    -----
    Diagonals are counted twice, since they are pairs of the same note
    and therefore each diagonal count contributes two copies of that
    note to the marginal frequency estimate.
    """
    diag = np.diagonal(mat)
    offdiag = np.sum(mat, axis=0) - diag
    # Diagonals are counted twice, since they are pairs of the same note
    frequency = offdiag + 2 * diag
    mutability = offdiag / frequency
    return mutability, frequency


######################################################################
### Process observations


def obs_to_dist_and_mat(observations):
    """
    Batch-convert a keyed collection of observation dictionaries into
    substitution-distance spectra and count matrices.

    Parameters
    ----------
    observations : dict
        Mapping from an arbitrary key (e.g. tune type, dataset label) to
        a substitution-count dictionary as produced by
        ``count_substitutions_from_msa``.

    Returns
    -------
    dist : dict
        Same keys as ``observations``; values are ``(X, Y)`` tuples
        returned by ``get_substition_distance``.
    matrices : dict
        Same keys as ``observations``; values are ``(letters, mat)``
        tuples returned by ``convert_observations_to_matrix``.
    """
    dist = {}
    matrices = {}
    for k, obs in observations.items():
        dist[k] = get_substition_distance('', obs=obs)
        matrices[k] = convert_observations_to_matrix(obs)
    return dist, matrices


def obs_mat_to_log_odds(mat):
    """
    Convert a raw substitution count matrix into a log-odds score matrix
    analogous to a BLOSUM-style scoring matrix.

    The log-odds score for substituting note i with note j is
    ``log(p_ij / (f_i * f_j * s_ij))``, where ``p_ij`` is the observed
    pair probability, ``f_i`` and ``f_j`` are the marginal note
    frequencies, and ``s_ij`` is 2 for off-diagonal entries (to account
    for symmetry) and 1 for diagonal entries.

    Parameters
    ----------
    mat : numpy.ndarray, shape (N, N)
        Symmetric substitution count matrix (upper triangle + diagonal
        are meaningful; lower triangle should mirror upper).

    Returns
    -------
    log_odds : numpy.ndarray, shape (N, N)
        Log-odds substitution scores. Cells where either the observed or
        expected probability is zero (i.e. non-finite log values) are set
        to ``np.nan``.

    Notes
    -----
    The derivation follows the BLOSUM methodology adapted for pitch-class
    substitutions:

    1. **Observed pair probabilities** (``prob``): the raw counts are
       normalised by the total number of unique unordered pairs, which is
       the sum of the upper triangle including the diagonal
       (``np.triu_indices(N, 0)``). This ensures the probability is
       computed over the same combinatorial space as the counts.

    2. **Marginal note frequencies** (``base_prob``): for note i,
       ``f_i = diag(prob)[i] + sum_j prob[i,j]``. The diagonal term is
       added once extra because a same-note pair (i, i) contributes two
       copies of note i to the marginal frequency (analogous to counting
       each sequence in a same-note column separately).

    3. **Expected pair probabilities** (``exp_prob``): under the null
       hypothesis of independent note usage, the probability of observing
       pair (i, j) is ``f_i * f_j`` for i == j (same-note), and
       ``2 * f_i * f_j`` for i != j (reflecting that either sequence
       could carry either note — the factor of 2 accounts for the
       symmetry of unordered pairs).

    4. **Log-odds**: ``log(prob) - log(exp_prob)`` gives the log-odds
       ratio; positive values indicate that the pair is more common than
       expected under independence (a conserved or preferred
       substitution), and negative values indicate it is rarer than
       expected.
    """
    # Convert counts of pairs to probabilities of pairs
    # (matrix is symmetric about the diagonal, so we only
    #  count one side, plus the diagonal)
    # Normalise by the upper-triangle count sum to get p_ij values that
    # are consistent with an unordered-pair probability model.
    prob = mat / np.sum(mat[np.triu_indices(len(mat), 0)])

    # To get the base probabilities of each note, we need
    # to count the diagonal twice, and everything else in the row once
    # Adding np.diag(prob) once extra reflects that a same-note pair at
    # position (i,i) contributes note i to the marginal frequency twice.
    base_prob = np.diag(prob) + np.sum(prob, axis=0)

    # The expected substitution probabilities is the outer
    # product of the base probabilities
    exp_prob = np.outer(base_prob, base_prob)

    # Double the off-diagonal elements, to reflect the symmetry
    # (A->G is the same as G->A)
    # Multiplying by (2 - I) doubles every off-diagonal entry while
    # leaving the diagonal unchanged (2-1=1 on the diagonal, 2-0=2 off).
    exp_prob *= (2 - np.eye(len(mat)))

#   # Get expected substitution probabilities from base probabilies
#   exp_prob = np.outer(np.diag(prob), np.diag(prob))
#   # Make off-diagonals equivalent
#   exp_prob[np.where(~np.eye(exp_prob.shape[0],dtype=bool))] *= 2

    # Get the log odds
    log_odds = np.log(prob) - np.log(exp_prob)
    # Cells with zero observed or expected probability produce -inf or NaN;
    # mask them so downstream code can handle missing scores explicitly.
    log_odds[~np.isfinite(log_odds)] = np.nan
    return log_odds


