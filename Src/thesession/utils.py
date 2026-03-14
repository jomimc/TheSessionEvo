from collections import Counter
import numpy as np
from scipy.stats import pearsonr, spearmanr

from thesession.config import *

######################################################################
### Convert sequence format


### Convert transposed chroma sequence to strings for alignment
def tchroma2seq(tchroma):
    try:
        return ''.join(letters[tchroma])
    except:
        return ''.join(letters[tchroma.astype(int)])


### Find where a substring starts and ends in a longer string
def find_matching_indices(string, substring):
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
    if isinstance(s1[0], str):
        if isinstance(s1, str):
            s1 = np.array(list(s1))
            s2 = np.array(list(s2))
        return np.where((s1 == '-')|(s2 == '-'))[0]
    return np.where(np.isnan(s1) | np.isnan(s2))


def get_corr(X, Y, p=0, s=0):
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
    diff = MODE_DIFF[(mode_old, mode_new)]
    for a, b in diff.items():
        tchroma[tchroma==a] = b
    return tchroma


def print_url(tune, setting=-1):
    if setting == -1:
        print(f"https://thesession.org/tunes/{tune}")
    else:
        print(f"https://thesession.org/tunes/{tune}#setting{setting}")


# Takes a list of lists of duration values and finds the smallest
# common denominator
def get_common_denominator(dur, tol=0.05):
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
    out = []
    dur_int = np.round(np.array(dur, float) * factor).astype(int)
    for t, n in zip(tc, dur_int):
        out.extend([t] * n)
    return np.array(out)


def inverse_frequency_weights(res, alpha):
    counts = res[['query_tunecount', 'target_tunecount']].values
    mean_tunecount = np.product(counts, axis=1)**0.5
    return mean_tunecount**(-alpha)


### Get indices of for separating tunes by mode
def get_mode_indices(res, mismatches, alg='exact'):
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
            if m in ['major', 'mixolydian']:
                i3 = res.query_mode == 'major pentatonic'
                i4 = res.target_mode == 'major pentatonic'
            else:
                i3 = res.query_mode == 'minor pentatonic'
                i4 = res.target_mode == 'minor pentatonic'
            idx_list.append(np.array((i1 & i2) | (i1 & i4) | (i2 & i3), bool))
    return idx_list





