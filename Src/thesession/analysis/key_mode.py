from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pearsonr, multinomial, entropy

from thesession.config import MODES, chromatic_map, chromatic_notes
from thesession.alignment import parts as PA
from thesession import utils


###################################################################
### Key/mode statistics and modal profiles


def get_key_mode_indices(target_mode):
    """
    Parse a mode string into integer indices for key and mode.

    Parameters
    ----------
    target_mode : str
        Mode string in the format "Cmajor", "C#minor", "Gdorian", etc.
        The tonic letter (with optional '#') is followed immediately by
        the mode name.

    Returns
    -------
    tuple of (int, int)
        ``(key_idx, mode_idx)`` where ``key_idx`` is the chromatic index
        of the tonic (0 = C, 1 = C#, ..., 11 = B) and ``mode_idx`` is the
        position of the mode name within the ordered ``MODES`` dict.

    Notes
    -----
    Relies on ``chromatic_map`` and ``MODES`` from ``thesession.config``.
    """
    # Build a name -> index lookup for the four supported modes
    modes = {m:i for i, m in enumerate(list(MODES.keys()))}
    if '#' in target_mode:
        # Sharp keys occupy two characters, e.g. "C#" before the mode name
        key_idx = chromatic_map[target_mode[:2]]
        mode_idx = modes[target_mode[2:]]
    else:
        # Natural keys are a single character
        key_idx = chromatic_map[target_mode[0]]
        mode_idx = modes[target_mode[1:]]
    return (key_idx, mode_idx)


def get_key_mode_priors(df, nkey=12, nmode=4):
    """
    Compute empirical prior probabilities over all key / mode combinations.

    Parameters
    ----------
    df : pandas.DataFrame
        Tune metadata; must contain a ``'mode'`` column with mode strings
        (e.g. "Cmajor").
    nkey : int, optional
        Number of chromatic pitch classes (default 12).
    nmode : int, optional
        Number of supported modes (default 4).

    Returns
    -------
    numpy.ndarray, shape (nkey, nmode)
        Normalised probability matrix.  Entry ``[i, j]`` is the fraction
        of tunes in key ``i`` and mode ``j``.

    Notes
    -----
    The prior is estimated from raw counts, so it reflects the actual
    distribution of keys and modes in the dataset rather than a uniform
    assumption.
    """
    # Convert each mode string to a (key_idx, mode_idx) integer pair
    key_mode_indices = df['mode'].apply(get_key_mode_indices).values
    count = Counter(key_mode_indices)
    prob = np.zeros((nkey, nmode), float)
    for (i, j), v in count.items():
        prob[i,j] = v
    # Normalise to obtain a proper probability distribution
    prob = prob / np.sum(prob)
    return prob


### Count tonal hierarchies for each mode
def get_modal_profiles(df, data):
    """
    Build empirical tonal-hierarchy (pitch-class histogram) profiles for
    each mode.

    Parameters
    ----------
    df : pandas.DataFrame
        Tune metadata; must contain columns ``'inferred_mode'``, ``'mode'``,
        ``'has_key_change'``, and ``'setting_id'``.
    data : dict
        Mapping from ``setting_id`` to a dict that contains at least
        ``'tchroma'`` — an array of transposed pitch classes (tonic = C = 0).

    Returns
    -------
    dict
        Keys are mode name strings (e.g. ``'major'``, ``'dorian'``); values
        are 1-D ``numpy.ndarray`` of length 12 giving the mean normalised
        pitch-class histogram across all qualifying tunes for that mode.

    Notes
    -----
    Three successive filters are applied before computing histograms:

    1. Only tunes whose ``inferred_mode`` is one of the four canonical modes
       (no pentatonic / mixed / indeterminate).
    2. Only tunes where the simple-algorithm annotation agrees with the
       dataset annotation, so that profiles are not contaminated by
       misclassified tunes.
    3. Only tunes without key changes, because a mid-tune modulation would
       blur the profile for the original mode.

    Each individual tune histogram is normalised to sum to 1 before
    averaging, so tunes of different lengths contribute equally.
    """
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
            # Normalise so every tune contributes equally regardless of length
            hist.append(h / h.sum())
        profiles[mode] = np.mean(hist, axis=0)
    return profiles


### Compute likelihood / correlations by comparing a tune's tonal hierarchy
### against all possible keys and modes
def profile_correlation(tchroma, mode_profiles, alg='bayesian'):
    """
    Score all 12 × 4 key/mode hypotheses against a tune's pitch-class
    histogram.

    This is the core key-finding routine.  For each of the 12 possible
    transpositions (i.e. candidate tonics) and each of the 4 modes, the
    tune's pitch-class histogram is aligned to the mode profile and a
    similarity score is computed.  Testing all 12 transpositions is
    achieved by cyclically rolling the observed histogram rather than
    re-transposing the raw pitch data.

    Parameters
    ----------
    tchroma : array-like of numeric
        Sequence of pitch classes for a single tune.  Values are reduced
        modulo 12 so octave information is discarded.  The tunes are
        assumed to be already transposed so that the tonic maps to C (0),
        but this function tests all 12 candidate tonics to find the best
        match.
    mode_profiles : dict
        Mapping from mode name string to a 1-D ``numpy.ndarray`` of length
        12 representing the expected normalised pitch-class histogram for
        that mode (tonic = index 0).  Typically the output of
        ``get_modal_profiles``.
    alg : {'bayesian', 'pearson'}, optional
        Scoring algorithm.

        * ``'bayesian'``: log-probability of the observed pitch-class
          counts under a multinomial distribution parameterised by the
          mode profile.  Yields an absolute likelihood comparable across
          tunes of different lengths.
        * ``'pearson'``: Pearson correlation between the mode profile and
          the (possibly rolled) observed histogram.  Scale-free but does
          not account for sample size.

    Returns
    -------
    numpy.ndarray, shape (12, n_modes)
        Score matrix.  Row ``i`` corresponds to transposition ``i``
        (i.e. the hypothesis that the tonic is pitch class ``i``); column
        ``j`` corresponds to the ``j``-th mode in ``MODES``.  Higher
        values indicate a better match.

    Notes
    -----
    Rolling the histogram by ``-i`` positions is equivalent to asking
    "what would the histogram look like if we relabelled pitch class ``i``
    as 0?", which is the same as transposing the tune down by ``i``
    semitones.  This avoids reprocessing the raw pitch sequence for every
    candidate key.

    The Bayesian branch uses ``scipy.stats.multinomial.pmf``.  When the
    observed histogram contains pitch classes that have zero probability
    under the profile, the PMF is 0 and the log becomes ``-inf``.  These
    entries are handled downstream by ``np.nanargmax`` or by filtering
    with ``np.isfinite``.

    The commented-out per-note mean log-probability variant
    (``np.mean(np.log(...))``) and the ``N**0.5`` length normalisation
    were explored but not adopted in the final analysis.
    """
#   if alg == 'bayesian':
#       tchroma = tchroma.astype(int)
#   elif alg == 'pearson':
    # Reduce to pitch classes 0–11 regardless of octave encoding
    tchroma = tchroma.astype(int) % 12
    bins = np.arange(-0.5, 12, 1)
    # Build the observed pitch-class histogram once; rolling it is cheaper
    # than recomputing it for each candidate transposition
    hist = np.histogram(tchroma, bins=bins)[0]
    N = len(tchroma)

    key_idx = np.arange(12)
    modes = list(MODES.keys())
    score = np.zeros((len(key_idx), len(modes)), float)
    for i in key_idx:
        for j, m in enumerate(modes):
            if alg == 'bayesian':
#               score[i,j] = np.mean(np.log(mode_profiles[m][(tchroma - i) % 12]))
                # Rolling by -i shifts the histogram so that pitch class i
                # aligns with index 0 of the mode profile, i.e. we test
                # the hypothesis that i is the tonic
                score[i,j] = np.log(multinomial.pmf(np.roll(hist, -i), N, mode_profiles[m]))#/ N**0.5
            elif alg == 'pearson':
                score[i,j] = pearsonr(mode_profiles[m], np.roll(hist, -i))[0]
    return score


### Compute the most likely key and mode for a tune
def assign_key_and_mode(tchroma, mode_profiles, alg='bayesian', priors=None):
    """
    Return the most likely key and mode for a tune.

    Parameters
    ----------
    tchroma : array-like of numeric
        Sequence of pitch classes (tonic = C = 0).
    mode_profiles : dict
        Mode profiles as returned by ``get_modal_profiles``.
    alg : {'bayesian', 'pearson'}, optional
        Scoring algorithm passed to ``profile_correlation``.
    priors : numpy.ndarray of shape (12, n_modes) or None, optional
        Log-prior probabilities over key/mode pairs.  If provided they are
        added to the log-likelihood scores before taking the argmax,
        implementing a MAP estimate.  Pass ``None`` (default) for a
        uniform prior (i.e. pure maximum-likelihood).

    Returns
    -------
    str
        Mode string for the best-scoring key/mode combination, e.g.
        ``'Cmajor'``, ``'G#dorian'``.

    Notes
    -----
    ``np.nanargmax`` is used so that ``-inf`` scores (which arise when a
    pitch class absent from the profile is observed) do not prevent
    other hypotheses from being selected.
    """
    modes = list(MODES.keys())
    score = profile_correlation(tchroma, mode_profiles, alg=alg)
    if not isinstance(priors, type(None)):
        # Add log-prior to convert log-likelihood into log-posterior
        score += np.log(priors)
    # Flatten the 2-D score matrix and recover the (key, mode) pair
    i, j = np.unravel_index(np.nanargmax(score), score.shape)
    return chromatic_notes[i] + modes[j]


def compute_tonal_ambiguity(tchroma, mode_profiles, priors=None):
    """
    Quantify how ambiguous the key/mode of a tune is.

    Parameters
    ----------
    tchroma : array-like of numeric
        Sequence of pitch classes (tonic = C = 0).
    mode_profiles : dict
        Mode profiles as returned by ``get_modal_profiles``.
    priors : numpy.ndarray of shape (12, n_modes) or None, optional
        Log-prior probabilities over key/mode pairs added to the
        log-likelihood scores before computing the posterior.

    Returns
    -------
    float
        Shannon entropy (in nats) of the normalised posterior distribution
        over key/mode hypotheses.  A value of 0 indicates a perfectly
        unambiguous key assignment; larger values indicate greater
        ambiguity.

    Notes
    -----
    Hypotheses with ``-inf`` log-scores (zero probability under every
    viable profile) are excluded via ``np.isfinite`` before normalising,
    so they do not distort the entropy calculation.
    """
    score = profile_correlation(tchroma, mode_profiles, alg='bayesian')
    if not isinstance(priors, type(None)):
        score += np.log(priors)
    # Discard -inf entries (zero-probability hypotheses) before normalising
    idx = np.isfinite(score)
    # Exponentiate to move from log-space to probability space, then normalise
    prob = np.exp(score[idx])
    return entropy(prob / prob.sum())


### Compute the likelihood/correlation score for a tune,
### given the correct key and mode
def score_key_and_mode(tchroma, mode_profiles, target_mode, alg='bayesian', priors=None):
    """
    Return the score for a specific key/mode hypothesis.

    Unlike ``assign_key_and_mode``, which picks the best hypothesis, this
    function looks up the score for the ground-truth (or otherwise
    specified) key and mode.  Useful for evaluating how confidently the
    correct answer is scored.

    Parameters
    ----------
    tchroma : array-like of numeric
        Sequence of pitch classes (tonic = C = 0).
    mode_profiles : dict
        Mode profiles as returned by ``get_modal_profiles``.
    target_mode : str
        Ground-truth mode string, e.g. ``'Cmajor'`` or ``'C#dorian'``.
    alg : {'bayesian', 'pearson'}, optional
        Scoring algorithm passed to ``profile_correlation``.
    priors : numpy.ndarray of shape (12, n_modes) or None, optional
        Not currently applied inside this function; reserved for future
        use or external combination.

    Returns
    -------
    float
        The score at position ``(key_idx, mode_idx)`` in the score matrix
        returned by ``profile_correlation``.
    """
    modes = {m:i for i, m in enumerate(list(MODES.keys()))}

    # Parse the target mode string to get matrix indices
    if '#' in target_mode:
        key_idx = chromatic_map[target_mode[:2]]
        mode_idx = modes[target_mode[2:]]
    else:
        key_idx = chromatic_map[target_mode[0]]
        mode_idx = modes[target_mode[1:]]

    score = profile_correlation(tchroma, mode_profiles, alg=alg)
    return score[key_idx, mode_idx]


def compute_tonal_ambiguity_family(res, parts, mode_profiles, tune_id, p0, meter, factor=4, pid=0.5, nran=10):
    """
    Compute tonal ambiguity for every member of a tune family as a
    function of note-subset size and conservation order.

    For each setting in the family the function measures how confidently
    the correct key can be inferred from subsets of notes selected by
    four strategies: the first N notes (temporal order), the N most
    conserved positions in the MSA, the N least conserved positions, and
    N randomly sampled notes.  This allows examination of whether
    tonally informative notes cluster at structurally important positions.

    Parameters
    ----------
    res : pandas.DataFrame
        Pairwise alignment results table as produced by
        ``PA.annotate_res``; used to retrieve family members.
    parts : dict
        Mapping from part identifier to part data (duration/pitch arrays).
    mode_profiles : dict
        Mode profiles as returned by ``get_modal_profiles``.
    tune_id : int or str
        Identifier of the tune family to analyse.
    p0 : int
        Part index within each setting to use (0-based).
    meter : str
        Time signature string, e.g. ``'6/8'`` or ``'4/4'``.  Passed to
        ``eval()`` to obtain the numeric value.
    factor : int, optional
        Rhythmic quantisation factor; the smallest note duration is mapped
        to ``1/factor`` of an eighth note (default 4).
    pid : float, optional
        Minimum sequence-identity threshold for including a setting in the
        MSA (default 0.5).
    nran : int, optional
        Number of random resamplings used to estimate the random-selection
        baseline at each note count (default 10).

    Returns
    -------
    collections.defaultdict
        Nested mapping ``out[part_id][strategy]`` where ``part_id`` is the
        part identifier string and ``strategy`` is one of:

        * ``'overall'`` — scalar entropy for the full melody.
        * ``'first'`` — ``numpy.ndarray`` of length 10 (N = 10, 20, …, 100).
        * ``'most_cons'`` — same shape; notes sorted by highest conservation.
        * ``'least_cons'`` — same shape; notes sorted by lowest conservation.
        * ``'random'`` — same shape; mean over ``nran`` random samples.

    Notes
    -----
    Conservation is measured by per-position Shannon entropy across the
    MSA (via ``PA.get_position_conservation``); low entropy means high
    conservation.  ``np.argsort(ent)`` therefore puts the most conserved
    positions first.

    The ``grid_per_bar`` computation uses a factor of 2 (not 4) because
    durations in the data are already stored in eighth-note units, so
    multiplying by ``2 * eval(meter)`` converts bars to eighth notes and
    then ``* factor`` converts to grid steps.
    """
    # Establish how many notes should be in a bar, given the factor used
    # to quantize the rhythm, and the meter
    grid_per_bar = int(2 * eval(meter) * factor)

    # Get the MSA for this tune family, and use it to compute the entropy along the sequence
    part_list, msa = PA.get_msa_family(res, parts, tune_id, p0, pid, grid_per_bar, factor=factor)
    ent = PA.get_position_conservation(msa)

    # Sort positions by sequence conservation (argsort of entropy gives
    # most-conserved first because low entropy = high conservation)
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
            # Take the first N notes in temporal order
            out[part]['first'][i] = compute_tonal_ambiguity(tc[:N], mode_profiles)
            # Take the N positions with the lowest positional entropy (most conserved)
            out[part]['most_cons'][i] = compute_tonal_ambiguity(tc[idx][:N], mode_profiles)
            # Reverse idx to put least-conserved positions first
            out[part]['least_cons'][i] = compute_tonal_ambiguity(tc[idx][::-1][:N], mode_profiles)
            # Average over nran random draws with replacement for a null baseline
            out[part]['random'][i] = np.mean([compute_tonal_ambiguity(np.random.choice(tc, size=N, replace=True), mode_profiles) for _ in range(nran)])
    return out


def predict_key_family(res, parts, mode_profiles, tune_id, p0, meter, factor=4, pid=0.5, nran=10):
    """
    Measure key-prediction accuracy for a tune family as a function of
    note-subset size and conservation order.

    Because all tunes in the dataset are transposed so the tonic is C
    (tchroma = 0), a correct key prediction is simply the one that
    returns ``'C'`` as the first character of the mode string.  The
    function compares four note-selection strategies (first-N, most
    conserved, least conserved, random) across a range of subset sizes.

    Parameters
    ----------
    res : pandas.DataFrame
        Pairwise alignment results table as produced by
        ``PA.annotate_res``.
    parts : dict
        Mapping from part identifier to part data (duration/pitch arrays).
    mode_profiles : dict
        Mode profiles as returned by ``get_modal_profiles``.
    tune_id : int or str
        Identifier of the tune family to analyse.
    p0 : int
        Part index within each setting to use (0-based).
    meter : str
        Time signature string, e.g. ``'6/8'`` or ``'4/4'``.  Passed to
        ``eval()`` to obtain the numeric value.
    factor : int, optional
        Rhythmic quantisation factor (default 4).
    pid : float, optional
        Minimum sequence-identity threshold for MSA inclusion (default 0.5).
    nran : int, optional
        Number of random resamplings for the random-selection baseline
        (default 10).

    Returns
    -------
    numpy.ndarray, shape (4, len(N_arr))
        Mean key-prediction accuracy averaged over all settings in the
        family.  Rows correspond to selection strategies:

        * 0 — first N notes (temporal order).
        * 1 — N most conserved positions.
        * 2 — N least conserved positions.
        * 3 — random sample of N notes (mean over ``nran`` draws).

        Columns correspond to the values in
        ``N_arr = [2, 4, 6, 8, 10, 15, 20, …, 50]``.
        Returns an all-NaN array of shape ``(4, len(N_arr))`` if no
        eligible settings are found for this family.

    Notes
    -----
    The boolean result of ``assign_key_and_mode(...)[0] == 'C'`` is cast
    to float implicitly when averaged, yielding a proportion in [0, 1].

    ``N_arr`` is constructed by concatenating a fine-grained range for
    very small note counts (2–8 in steps of 2) with a coarser range
    (10–50 in steps of 5).  This density at low N is intentional: key
    perception from very few notes changes rapidly and the fine sampling
    captures that transition.

    The commented-out alternative ``N_arr = np.arange(5, 55, 5)`` was
    superseded by the current construction to capture sub-10-note
    behaviour.
    """
    # Establish how many notes should be in a bar, given the factor used
    # to quantize the rhythm, and the meter
    grid_per_bar = int(2 * eval(meter) * factor)

#   N_arr = np.arange(5, 55, 5)
    # Fine resolution at low N, coarser above 10, to capture rapid changes
    # in key-prediction accuracy when only a handful of notes are available
    N_arr = np.concatenate([np.arange(2, 10, 2), np.arange(10, 55, 5)])

    # Get the MSA for this tune family, and use it to compute the entropy along the sequence
    part_list, msa = PA.get_msa_family(res, parts, tune_id, p0, pid, grid_per_bar, factor=factor)
    if len(part_list) == 0:
        # No eligible settings found; return a NaN placeholder of the expected shape
        return np.zeros((4, N_arr.size)) * np.nan

    # Get entropy (inverse of sequence convservation)
    ent = PA.get_position_conservation(msa)

    # Sort positions by sequence conservation (most conserved first)
    idx = np.argsort(ent)

    # Results container: axes are (settings, strategies, N_values)
    out = np.zeros((len(part_list), 4, N_arr.size), bool)

    for i, tc in enumerate(msa):
        for j, N in enumerate(N_arr):
            # Strategy 0: first N notes in temporal order
            out[i,0,j] = assign_key_and_mode(tc[:N], mode_profiles)[0] == 'C'
            # Strategy 1: N most conserved positions across the family MSA
            out[i,1,j] = assign_key_and_mode(tc[idx][:N], mode_profiles)[0] == 'C'
            # Strategy 2: N least conserved positions (reverse of sorted order)
            out[i,2,j] = assign_key_and_mode(tc[idx][::-1][:N], mode_profiles)[0] == 'C'
            # Strategy 3: mean accuracy over nran random samples with replacement
            out[i,3,j] = np.mean([assign_key_and_mode(np.random.choice(tc, size=N, replace=True), mode_profiles)[0] == 'C' for _ in range(nran)])
    # Average across settings (axis 0) to get family-level accuracy per strategy
    return out.mean(axis=0)


def compare_pearson_and_bayesian(mode_profiles, mode='major', nrep=100):
    """
    Empirically compare Bayesian and Pearson key-finding accuracy on
    synthetic pitch sequences.

    Parameters
    ----------
    mode_profiles : dict
        Mode profiles as returned by ``get_modal_profiles``.
    mode : str, optional
        The mode used to generate synthetic pitch sequences (default
        ``'major'``).  Pitches are sampled i.i.d. from the corresponding
        mode profile treated as a categorical distribution.
    nrep : int, optional
        Number of random repetitions per note-count value (default 100).

    Returns
    -------
    numpy.ndarray, shape (len(N_arr), nrep, 2)
        Boolean accuracy array.  The last axis indexes algorithm:
        index 0 = Bayesian, index 1 = Pearson.  Each entry is ``True``
        when the algorithm correctly predicts C as the tonic.

    Notes
    -----
    Because pitches are drawn from the mode profile with tonic = C,
    the ground-truth key is always C.  Accuracy is therefore the
    fraction of repetitions where the first character of the returned
    mode string equals ``'C'``.

    ``N_arr`` mirrors the one used in ``predict_key_family`` to make
    results directly comparable.
    """
    # Use the same note-count grid as predict_key_family for comparability
    N_arr = np.concatenate([np.arange(2, 10, 2), np.arange(10, 55, 5)])
    res = []
    for N in N_arr:
        for _ in range(nrep):
            # Draw N pitches i.i.d. from the mode profile; ground truth is C
            pitch_vec = np.random.choice(np.arange(12), size=N, replace=True, p=mode_profiles[mode])
            res.append([assign_key_and_mode(pitch_vec, mode_profiles)[0] == 'C',
                        assign_key_and_mode(pitch_vec, mode_profiles, alg='pearson')[0] == 'C'])
    return np.array(res).reshape(N_arr.size, nrep, 2)
