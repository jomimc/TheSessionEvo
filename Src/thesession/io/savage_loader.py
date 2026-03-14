from collections import Counter, defaultdict
from pathlib import Path
import pickle

import music21
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from thesession.config import *
from thesession.io import tune_loader as load_tunes
from thesession.analysis import optimization as OP
from thesession.alignment import pairwise as seq_align
from thesession.analysis import substitution as SM
from thesession.io import tune_parser as TP
from thesession import utils


### Load the set of manually-aligned sequence pairs from Savage et al.
def load_savage_df(full=False, redo=False, keep_japanese=False):
    if full:
        path = PATH_BASE.joinpath("Results/savage_full.pkl")
    else:
        path = PATH_BASE.joinpath("Results/savage_part.pkl")

    if path.exists() and not redo:
        return pd.read_pickle(path)

    if full:
        df = pd.read_excel('../Data/Bronson/MelodicEvoSeqFullSongs.xlsx', sheet_name='MelodicEvoSeq')
    else:
        df = pd.read_excel('../Data/Bronson/MelodicEvoSeq.xlsx', sheet_name='MelodicEvoSeq')
        if not keep_japanese:
            df = df.loc[df.Language=='English']

    # Rename columns and drop unwanted columns
    old_col = ['PairNo', 'Song title', 'Language', 'Child Ballad no./NHK Volume no.',
               'Variant no.', 'PID', 'Full note sequence (aligned)', 'Full note sequence (unaligned)']
    new_col = ['PairNo', 'name', 'Language', 'chapter', 'song_ref', 'PID', 'seq_aligned', 'seq_unaligned']
    df = df.loc[:, old_col]
    df = df.rename(columns={c1:c2 for c1, c2 in zip(old_col, new_col)})

    # Remove Addenda names ('App') from chapters
    # And converts whitespace to underscore (necessary for mmseqs)
    if full:
        df['chapter'] = df['chapter'].apply(lambda x: x.replace('App', '').replace(" ", "_") if isinstance(x, str) else str(x))

    # Create a unique reference field 
    df['ref'] = [f"{str(c)}_{str(s)}" for c, s in zip(df['chapter'], df['song_ref'])]
    
    # Convert Pat's codings to integer values
    idx2drop = [] # Some of the codings in Savage et al are wrong, so keep a list of these
    tchroma = []
    for i, s in zip(df.index, df['seq_unaligned']):
        try:
            tchroma.append(np.array([note_map[n] for n in s.strip()]))
        except:
            idx2drop.append(i)
    df = df.drop(index=idx2drop)
    df['tchroma'] = tchroma

    # Save to pickle
    df.to_pickle(path)

    return df


def load_checked_subset():
    dfb = load_tunes.load_bronson_data()
    dfs = load_savage_df()

    df = pd.read_csv('../savage_matched_songs.csv')
    df = df.loc[df.IndexFitch.notnull()]
    df['IndexFitch'] = df['IndexFitch'].astype(int)

    df['tchroma_savage'] = dfs.loc[df.Index, 'tchroma'].values

    key = dfb.loc[df.IndexFitch, 'key']
    midi = []
    for i in df.IndexFitch:
        data = dfb.loc[i].to_dict()
        data['key'] = chromatic_notes[int(data['key'])]
        midi.append(TP.parse_thesession_tune(data, 'music21', expandRepeats=False)['midi'])
    df['tchroma_fitch'] = [((np.array(m) - k) % 12).astype(int) for m, k in zip(midi, key)]

    df['len_same'] = [len(x) == len(y) for x, y in zip(df['tchroma_savage'], df['tchroma_fitch'])]
    df['exact_same'] = [len(x) == len(y) and np.all(x == y) for x, y in zip(df['tchroma_savage'], df['tchroma_fitch'])]

    return df


### Load the full set of sequence pairs from Savage et al.
def load_df_full():
    return pd.read_excel('../Data/Bronson/MelodicEvoSeqFullSongs.xlsx', sheet_name='MelodicEvoSeq')


### Given a sequence, and an aligned version (i.e. the sequence plus gaps),
### replace the characters in the aligned version with the characters from
### the given sequence
def convert_seq(tchroma, aligned):
    gaps = np.where(np.array(list(aligned)) == '-')[0]
    seq = ''.join(utils.tchroma2seq(tchroma))
    for i in gaps:
        seq = seq[:i] + '-' + seq[i:]
    return seq


### For each sequence pair in Savage et al, run local alignment algorithm,
### and check if the manually-aligned alignments are one of the top-scoring
### alignments
def compare_all_alignments(df, **kwargs):
    pair_list = np.array(sorted(df['PairNo'].unique()))
    is_found = []
    num_align = []
    for pair in pair_list:
        idx = df.loc[df['PairNo'] == pair].index
        al1, al2 = [convert_seq(*df.loc[i, ['tchroma', 'seq_aligned']]) for i in idx]
        s1, s2 = [utils.tchroma2seq(df.loc[i, 'tchroma']) for i in idx]
        alignments = seq_align.get_pairwise_nhits(s1, s2, **kwargs)
#       print(f"Pair {pair}. Checking {len(alignments)} alignments...")
        is_found.append(is_alignment_found(al1, al2, alignments))
        num_align.append(len(alignments))
    return np.array(is_found), np.array(num_align)



### Given a set of alignments, check if it contains a specific alignment (al1, al2).
### To save time, only look at the first 200 alignments in a set (since there can
### be hundreds of thousands in some cases...)
def is_alignment_found(al1, al2, alignments, max_align=200):
    for i, aln in enumerate(alignments):
        if i >= max_align:
            break
        if all([a == b for a, b in zip(aln, [al1, al2])]):
            return True
    return False


### Load results for algorithm that checks whether manual alignments
### are within the best-scoring algorithmic alignments
def alignment_results():
    df = load_savage_df()
    dfr, freq_correct = OP.load_results_savage()
    df['freq_correct'] = np.vstack([freq_correct]*2).T.ravel()
    print(f"Percentage of pairs that never align properly: {np.mean(df.freq_correct==0)}")
    num_align = np.load(dfr.loc[2869, 'path'])[1]
    df['num_align'] = np.vstack([num_align]*2).T.ravel()
    correct = np.load(dfr.loc[2869, 'path'])[0]
    df['is_correct'] = np.vstack([correct]*2).T.ravel()
#   np.mean(df.loc[(df.is_correct==1), 'num_align'] == 1)
#   Counter(df.loc[(df.is_correct==1)&(df['num_align'] == 1), 'num_gaps'])


def get_submat(df):
    obs = defaultdict(float)
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
        idx1 = np.where(al1 != '-')[0]
        idx2 = np.where(al2 != '-')[0]
        tc1 = tc1[al2[idx1] != '-']
        tc2 = tc2[al1[idx2] != '-']

        # Get substitutions
        sub_idx = np.where(tc1 != tc2)[0]
        for i in sub_idx:
            obs[(tc1[i], tc2[i])] += 1

        # Get aligned notes
        same_idx = np.where(tc1 == tc2)[0]
        for k, v in Counter(tc1[same_idx]).items():
            obs[(k, k)] += v
    letters, mat = SM.convert_observations_to_matrix(obs, True)
        
    return obs, letters, mat


