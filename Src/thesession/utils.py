from collections import Counter
import numpy as np
from scipy.stats import pearsonr, spearmanr

from thesession.config import *

######################################################################
### Convert sequence format


### Convert transposed chroma sequence to strings for alignment
def tchroma2seq(tchroma):
    """
    Convert an integer pitch-class sequence to a protein-letter string.

    The ``letters`` array (defined in ``config.py``) maps each integer
    index to a single character from the IUPAC protein alphabet, giving
    a string representation that can be fed directly to sequence-alignment
    tools such as MMseqs2 or Biopython's ``PairwiseAligner``.

    Parameters
    ----------
    tchroma : array-like of int
        Pitch-class values in [0, 11].

    Returns
    -------
    str
        Single string of protein letters corresponding to ``tchroma``.

    Notes
    -----
    The ``try``/``except`` block handles two common input types: integer
    numpy arrays (direct indexing works) and float arrays (which must be
    cast to ``int`` first).
    """
    try:
        return ''.join(letters[tchroma])
    except:
        return ''.join(letters[tchroma.astype(int)])


### Find where a substring starts and ends in a longer string
def find_matching_indices(string, substring):
    """
    Locate the start and end positions of ``substring`` within ``string``.

    Parameters
    ----------
    string : str
        The full string to search in.
    substring : str
        The substring to locate.

    Returns
    -------
    start : int
        Index of the first character of ``substring`` in ``string``.
    end : int
        Index one past the last character (i.e. ``start + len(substring)``).

    Notes
    -----
    If ``substring`` is not found, a warning is printed and ``None`` is
    returned implicitly.  The caller should guard against this case.
    """
    N = len(substring)
    for i in range(len(string) - N + 1):
        if string[i:i+N] == substring:
            return i, i + N
    print ("Error! substring not identified!")


### In principle this should be made general, so that:
###     tchroma should be the full sequence, that can be converted to letters
###     align should be the letters in the alignment
###     tpich_oct should be any sequence
def reverse_mapping(tchroma, tchroma_oct, align):
    """
    Project a secondary sequence (e.g. octave-relative pitch) back onto
    an alignment array, respecting gap positions and possible tail truncation.

    Given a full pitch-class sequence ``tchroma`` and a parallel sequence
    of a different representation ``tchroma_oct``, and an alignment array
    ``align`` (which may contain gap characters ``'-'`` and may cover only
    a sub-sequence of ``tchroma``), this function returns an array of the
    same length as ``align`` where non-gap positions are filled with the
    corresponding values from ``tchroma_oct``.

    Parameters
    ----------
    tchroma : array-like of int
        Full pitch-class sequence for the tune (length = total notes).
    tchroma_oct : array-like of int or str
        Parallel representation to map (e.g. octave-transposed pitch).
        Must be the same length as ``tchroma``.
    align : numpy.ndarray
        Alignment row for this sequence, possibly containing ``'-'`` gap
        characters.  Non-gap characters must match a contiguous substring
        of ``letters[tchroma]``.

    Returns
    -------
    new_align : numpy.ndarray
        Array with the same shape and dtype as ``align`` (``str`` or
        ``float``), with gap positions filled with ``'-'`` or ``np.nan``
        respectively and non-gap positions filled from ``tchroma_oct``.

    Notes
    -----
    If the number of non-gap positions equals the full length of
    ``tchroma``, the mapping is applied directly.  Otherwise the function
    locates the matching sub-sequence via ``find_matching_indices`` and
    slices ``tchroma_oct`` accordingly (handling tail-truncated alignments).
    """
    # get the full tchroma string, converted to letters, for comparison with the msa
    letter_seq = ''.join(letters[tchroma])
    # get the indices without gaps
    idx_nongap = align != '-'
    # Create copy, so original is not modified
    dtype = str if isinstance(tchroma_oct[0], str) else float
    new_align = np.zeros_like(align, dtype=dtype)
    if dtype == str:
        new_align[:] = '-'
    else:
        new_align[:] = np.nan

    # if the tails are not truncated...
    if len(letter_seq) == np.sum(idx_nongap):
        new_align[idx_nongap] = tchroma_oct
    # if the tails are truncated...
    else:
        start, end = find_matching_indices(letter_seq, ''.join(align[idx_nongap]))
        new_align[idx_nongap] = tchroma_oct[start:end]
    return new_align


def reverse_mapping_idx(tchroma, align):
    """
    Return the start and end indices into ``tchroma`` that correspond to
    the non-gap portion of an alignment row.

    Parameters
    ----------
    tchroma : array-like of int
        Full pitch-class sequence for the tune.
    align : numpy.ndarray
        Alignment row for this sequence, possibly containing ``'-'`` gap
        characters.

    Returns
    -------
    start : int
        Index of the first note of the aligned sub-sequence in ``tchroma``.
    end : int
        Index one past the last note (exclusive).

    Notes
    -----
    If the alignment covers the complete sequence (no tail truncation),
    ``(0, len(tchroma))`` is returned directly without a substring search.
    """
    # get the full tchroma string, converted to letters, for comparison with the msa
    letter_seq = ''.join(letters[tchroma])
    # get the indices without gaps
    idx_nongap = align != '-'

    # if the tails are not truncated...
    if len(letter_seq) == np.sum(idx_nongap):
        return 0, len(letter_seq)
    # if the tails are truncated...
    else:
        start, end = find_matching_indices(letter_seq, ''.join(align[idx_nongap]))
        return start, end


def pairwise_reverse_mapping(df, res, pairwise_align, ref='setting_id'):
    """
    Apply ``reverse_mapping`` to every pairwise alignment, replacing
    letter-encoded pitch classes with octave-relative pitch values.

    Parameters
    ----------
    df : pandas.DataFrame
        Metadata table with at least two columns: ``ref`` (sequence
        identifier) and ``tchroma`` / ``tchroma_octave`` (pitch-class and
        octave-relative pitch arrays for each sequence).
    res : pandas.DataFrame
        Hit table with columns ``query`` and ``target`` (sequence IDs).
    pairwise_align : list of array-like
        List of length ``len(res)``; each element is a 2-element sequence
        ``[align_query, align_target]`` where each alignment row is a
        numpy string array of letters and ``'-'`` gap characters.
    ref : str, optional
        Name of the column in ``df`` to use as the sequence identifier.
        Default is ``'setting_id'``.

    Returns
    -------
    pairwise_oct : list of numpy.ndarray, shape (2, alignment_length)
        One array per hit pair; each array has the octave-relative pitch
        values (or ``np.nan`` at gap positions) for the query (row 0)
        and target (row 1).
    """
    pairwise_oct = []
    setting2tchroma = {s: tp for s, tp in zip(df[ref], df.tchroma)}
    setting2tchroma_oct = {s: tp for s, tp in zip(df[ref], df.tchroma_octave)}
    for i, (s1, s2) in enumerate(zip(*res.loc[:,['query', 'target']].values.T)):
        new_align = []
        for j, s in enumerate([s1, s2]):
            tchroma = setting2tchroma[s]
            tchroma_oct = setting2tchroma_oct[s]
            msa = pairwise_align[i][j].copy().astype("U3")
            new_align.append(reverse_mapping(tchroma, tchroma_oct, msa))
        pairwise_oct.append(np.array(new_align))
    return pairwise_oct


def find_gaps(s1, s2):
    """
    Return indices of positions where at least one of two aligned sequences
    contains a gap or missing value.

    Parameters
    ----------
    s1 : numpy.ndarray or str
        First aligned sequence (letter array or string, or float array).
    s2 : numpy.ndarray or str
        Second aligned sequence with the same length as ``s1``.

    Returns
    -------
    numpy.ndarray of int
        Indices where ``s1 == '-'``, ``s2 == '-'``, ``np.isnan(s1)``, or
        ``np.isnan(s2)``.

    Notes
    -----
    String sequences (including raw Python strings) are detected by
    checking the type of the first element.  Float sequences use
    ``np.isnan`` as the missing-value sentinel.
    """
    if isinstance(s1[0], str):
        if isinstance(s1, str):
            s1 = np.array(list(s1))
            s2 = np.array(list(s2))
        return np.where((s1 == '-')|(s2 == '-'))[0]
    return np.where(np.isnan(s1) | np.isnan(s2))


def get_corr(X, Y, p=0, s=0):
    """
    Compute the correlation between two arrays, ignoring non-finite values.

    Parameters
    ----------
    X : numpy.ndarray
        First variable.
    Y : numpy.ndarray
        Second variable, same length as ``X``.
    p : int, optional
        If non-zero, return the full ``(statistic, p-value)`` tuple.
        Default is ``0`` (return statistic only).
    s : int, optional
        If non-zero, use Spearman rank correlation; otherwise use Pearson.
        Default is ``0`` (Pearson).

    Returns
    -------
    float or tuple
        Correlation coefficient, or ``(coefficient, p-value)`` if
        ``p != 0``.
    """
    idx = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[idx], Y[idx]
    fn = spearmanr if s else pearsonr
    if p:
        return fn(X, Y)
    else:
        return fn(X, Y)[0]


### This doesn't account for ambiguity, in cases where there
### are equal amounts of major/minor thirds, or 6ths/7ths,
### in which case the mode is not clear
def check_mode(tchroma):
    """
    Infer the musical mode of a tune from its pitch-class histogram.

    The mode is determined by comparing the counts of characteristic
    scale degrees: major vs. minor third (degrees 4 vs. 3), major vs.
    minor sixth (degrees 9 vs. 8), and major vs. minor seventh (degrees
    11 vs. 10).

    Parameters
    ----------
    tchroma : array-like of int
        Pitch-class sequence transposed to the tonic (0 = tonic).

    Returns
    -------
    str
        One of ``'major'``, ``'mixolydian'``, ``'minor'``, ``'dorian'``,
        ``'major pentatonic'``, ``'minor pentatonic'``,
        ``'mixolydian/dorian'``, ``'minor/dorian'``, or
        ``'indeterminate'``.

    Notes
    -----
    This function does not account for ambiguity when discriminating
    intervals appear in equal counts (e.g. equal major and minor thirds).
    Such cases fall through to the ``'minor/dorian'`` or
    ``'indeterminate'`` branches.
    """
    count = Counter(tchroma)
    mi3, ma3, mi6, ma6, mi7, ma7 = [count.get(x,0) for x in [3, 4, 8, 9, 10, 11]]
    # Major vs Minor
    if ma3 > mi3:
        # Major vs Mixolydian
        if ma7 > mi7:
            return 'major'
        elif ma7 < mi7:
            return 'mixolydian'
        elif ma7 + mi7 == 0:
            return 'major pentatonic'
        else:
            return 'minor/dorian'
    elif ma3 < mi3:
        # Minor vs Dorian
        if mi6 > ma6:
            return 'minor'
        elif mi6 < ma6:
            return 'dorian'
        elif mi6 + ma6 == 0:
            return 'minor pentatonic'
        else:
            return 'minor/dorian'

    # If no thirds, we can still potentilly tell apart major and minor,
    # but the other two modes are indistinguishable
    else:
        if (ma6 > 0) & (ma7 > 0) & (mi6 == 0) & (mi7 == 0):
            return 'major'
        elif (ma6 > 0) & (mi7 > 0) & (mi6 == 0) & (ma7 == 0):
            return 'mixolydian/dorian'
        elif (mi6 > 0) & (mi7 > 0) & (ma6 == 0) & (ma7 == 0):
            return 'minor'
        else:
            return 'indeterminate'


def change_mode(tchroma, mode_old, mode_new):
    """
    Remap the characteristic scale degrees of a pitch-class sequence from
    one mode to another.

    Parameters
    ----------
    tchroma : numpy.ndarray of int
        Pitch-class sequence to transform (modified in place).
    mode_old : str
        Source mode name (key into ``MODE_DIFF``).
    mode_new : str
        Target mode name (key into ``MODE_DIFF``).

    Returns
    -------
    tchroma : numpy.ndarray of int
        The modified array (same object as input).

    Notes
    -----
    ``MODE_DIFF[(mode_old, mode_new)]`` is a dict mapping each differing
    scale degree in ``mode_old`` to its counterpart in ``mode_new``.
    Only the degrees that differ between the two modes are changed;
    shared degrees are left untouched.
    """
    diff = MODE_DIFF[(mode_old, mode_new)]
    for a, b in diff.items():
        tchroma[tchroma==a] = b
    return tchroma


def print_url(tune, setting=-1):
    """
    Print the TheSession URL for a tune or a specific setting.

    Parameters
    ----------
    tune : int
        Tune ID on TheSession website.
    setting : int, optional
        Setting ID.  If ``-1`` (default), print the tune-level URL
        without a setting anchor.

    Returns
    -------
    None
    """
    if setting == -1:
        print(f"https://thesession.org/tunes/{tune}")
    else:
        print(f"https://thesession.org/tunes/{tune}#setting{setting}")


# Takes a list of lists of duration values and finds the smallest
# common denominator
def get_common_denominator(dur, tol=0.05):
    """
    Find the smallest integer factor that maps all note durations to
    integers (within a tolerance).

    This is used to determine the grid resolution needed to represent
    a set of note durations as integer numbers of time slots.

    Parameters
    ----------
    dur : list of array-like
        Each element is an array of note duration values (in eighth-note
        units) for one part.  All arrays are pooled together when
        searching for the common denominator.
    tol : float, optional
        Maximum allowed deviation from an integer when testing each
        candidate factor.  Default is ``0.05``.

    Returns
    -------
    int
        The smallest factor from the candidate list for which
        ``dur_value * factor`` is within ``tol`` of an integer for every
        unique duration value.  Returns ``0`` if no suitable factor is
        found.

    Notes
    -----
    The candidate factors ``[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 60,
    64]`` cover binary, ternary, and quinternary subdivisions at the
    ranges found in TheSession data. The vectorised outer-product test
    checks all candidates simultaneously and picks the smallest passing
    one.
    """
    # Get unique duration values
    vals = np.unique([float(x) for y in dur for x in y])

    # These factors should work for 2 and 3 and 5 subdivisions,
    # at least for the range found in thesession tunes
    factors = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 60, 64])

    # Multiply duration values by integer factors, and round
    prod = np.outer(vals, factors)
    int_prod = np.round(prod)

    # Get the indices of factors for which the products are sufficiently
    # close to integers
    is_close = np.where(np.all(np.abs(prod - int_prod) < tol, axis=0))[0]

    # If unsuccessful, return a zero
    if len(is_close) == 0:
        return 0

    # Otherwise return the smallest factor
    return factors[is_close[0]]


def get_tchroma_grid(tc, dur, factor):
    """
    Expand a pitch-class sequence onto a regular time grid.

    Each note is repeated for a number of grid positions equal to
    ``round(dur[i] * factor)``, producing a time-indexed array where
    every cell holds the pitch class of the note sounding at that instant.

    Parameters
    ----------
    tc : array-like of int
        Pitch-class values (e.g. ``tmidi % 12``), one per note.
    dur : array-like of float
        Note durations in eighth-note units, parallel to ``tc``.
    factor : int or float
        Grid resolution multiplier; ``dur[i] * factor`` gives the number
        of grid slots occupied by note ``i``.

    Returns
    -------
    numpy.ndarray of int
        1-D array of length ``sum(round(dur[i] * factor))`` containing
        the pitch class repeated for each grid slot.
    """
    out = []
    # Convert floating-point durations to integer grid slot counts
    dur_int = np.round(np.array(dur, float) * factor).astype(int)
    for t, n in zip(tc, dur_int):
        out.extend([t] * n)
    return np.array(out)


def inverse_frequency_weights(res, alpha):
    """
    Compute inverse-frequency weights that down-weight tune pairs
    involving prolific tunes.

    Parameters
    ----------
    res : pandas.DataFrame
        Hit table with columns ``query_tunecount`` and
        ``target_tunecount``, where each value is the total number of
        hit-table appearances of that tune.
    alpha : float
        Exponent controlling the strength of down-weighting.  ``alpha=0``
        gives uniform weights; ``alpha=1`` gives full inverse-count
        weighting; ``alpha=0.5`` (square-root) is the default used
        throughout the pipeline.

    Returns
    -------
    numpy.ndarray of float
        1-D array of weights, one per row of ``res``.  The weight for a
        pair is the geometric mean of the two tune counts raised to the
        power ``-alpha``.

    Notes
    -----
    The geometric mean ``sqrt(count_query * count_target)`` is used
    rather than the arithmetic mean so that pairs of two very common
    tunes receive the most aggressive down-weighting.
    """
    counts = res[['query_tunecount', 'target_tunecount']].values
    # Geometric mean of the two tune counts
    mean_tunecount = np.product(counts, axis=1)**0.5
    return mean_tunecount**(-alpha)


### Get indices of for separating tunes by mode
def get_mode_indices(res, mismatches, alg='exact'):
    """
    Build per-mode boolean index arrays for filtering a hit table.

    Parameters
    ----------
    res : pandas.DataFrame
        Filtered hit table with columns ``query_mode`` and
        ``target_mode``.
    mismatches : pandas.DataFrame
        Per-pair alignment statistics (same index as ``res``; not
        currently used but kept for API consistency).
    alg : str, optional
        Matching algorithm.  One of:

        ``'exact'``
            Both query and target must have exactly the same mode string.
        ``'loose'``
            Mode label is treated as a substring match (e.g.
            ``'minor'`` matches ``'minor pentatonic'``).
        ``'exact_pent'``
            Like ``'exact'`` but also accepts pentatonic variants as
            compatible with their parent mode (major pentatonic ↔ major
            and mixolydian; minor pentatonic ↔ minor and dorian).
        ``'loose_pent'``
            Like ``'loose'`` with the same pentatonic inclusions.

        Default is ``'exact'``.

    Returns
    -------
    idx_list : list of numpy.ndarray of bool
        One boolean mask per mode in ``MODES`` (in insertion order:
        major, mixolydian, minor, dorian).  ``idx_list[k][i]`` is
        ``True`` if hit ``i`` should be included in the analysis for
        mode ``k``.

    Notes
    -----
    For ``'exact_pent'`` and ``'loose_pent'``, a hit is included for
    mode *m* if both ends are mode *m* (or its pentatonic variant), or
    if one end is exactly mode *m* and the other is the corresponding
    pentatonic.  The three inclusion conditions are OR-ed:
    ``(m & m) | (m & pent) | (pent & m)``.
    """
    if alg == 'exact':
        idx_list = [np.array((res.target_mode == res.query_mode) & (res.target_mode==m), bool) for m in MODES.keys()]
    elif alg == 'loose':
        idx_list = [np.array((res.query_mode.apply(lambda x: m in x)) &
                             (res.target_mode.apply(lambda x: m in x)), bool) for m in MODES.keys()]
    else:
        idx_list = []
        for m in MODES.keys():
            if alg == 'exact_pent':
                i1 = res.query_mode == m
                i2 = res.target_mode == m
            elif alg == 'loose_pent':
                i1 = res.query_mode.apply(lambda x: m in x)
                i2 = res.target_mode.apply(lambda x: m in x)
            # Pentatonic variant is 'major pentatonic' for major/mixolydian,
            # 'minor pentatonic' for minor/dorian
            if m in ['major', 'mixolydian']:
                i3 = res.query_mode == 'major pentatonic'
                i4 = res.target_mode == 'major pentatonic'
            else:
                i3 = res.query_mode == 'minor pentatonic'
                i4 = res.target_mode == 'minor pentatonic'
            idx_list.append(np.array((i1 & i2) | (i1 & i4) | (i2 & i3), bool))
    return idx_list
