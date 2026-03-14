import json
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesession.config import *
from thesession.io import tune_parser as TP
from thesession import utils



######################################################################
### thesession data


### Main function to load thesession data
### Runs preliminary pipeline if there is no cached data,
### or if redo=True
def load_thesession_data(redo=False):
    path_df = PATH_BASE.joinpath("Cache", "thesession_cleaned_processed.pkl")
    path_data = PATH_BASE.joinpath("Cache", "thesession_music21.pkl")
    if (path_df.exists() and path_data.exists()) and not redo:
        df = pd.read_pickle(path_df)
        data = pickle.load(open(path_data, 'rb'))
        return df, data
    else:
        return process_thesession_tunes(redo)


### Load the raw data from thesession downloaded from github
def load_thesession_data_raw():
    df = pd.read_csv(PATH_DATA.joinpath("TheSession-data", "csv", "tunes.csv"))
    json_data = json.load(open(PATH_DATA.joinpath("TheSession-data", "json", "tunes.json"), 'r'))
    return df, json_data


### Process all of thesession data, to produce a final version
def process_thesession_tunes(redo=False):
    # Load raw data
    df, json_data = load_thesession_data_raw()

    # Process with pyabc
    df = process_thesession_tunes_pyabc(df, json_data, redo=False, full=False)

    # Process with music21
    tunes = process_thesession_tunes_music21(df, json_data, redo=False)

    # Update the music21 data
    tunes = update_music21_data(df, tunes)

    # Remove all tunes that music21 did not parse
    df = df.loc[df.setting_id.apply(lambda x: x in tunes)]
    print(f"Tunes processed by music21. {len(df)} tunes left after cleaning...")

    # Statistically infer the most likely mode, given the tonal hierachy
    df['inferred_mode'] = df['setting_id'].apply(lambda x: utils.check_mode(tunes[x]['tchroma']))

    path_df = PATH_BASE.joinpath("Cache", "thesession_cleaned_processed.pkl")
    path_data = PATH_BASE.joinpath("Cache", "thesession_music21.pkl")
    df.to_pickle(path_df)
    pickle.dump(tunes, open(path_data, 'wb'))

    return df, tunes


### Primary processing of thesession data, using pyabc and custom code
### This stage is required for filtering out problematic tunes
def process_thesession_tunes_pyabc(df, json_data, redo=False, full=False):
    if full:
        path = PATH_BASE.joinpath("Cache", "thesession_full.pkl")
    else:
        path = PATH_BASE.joinpath("Cache", "thesession_clean.pkl")

    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        print(f"Starting to parse TheSession. {len(df)} tunes to start with...")
        parsed_data = TP.parse_thesession_tune(json_data[0])
        for k in parsed_data.keys():
            df[k] = pd.Series(dtype=object)

        idx_keep = []
        idx_ignored = []
        for i, data in tqdm(zip(df.index, json_data)):
            try:
                parsed_data = TP.parse_thesession_tune(data)
                df.loc[i, list(parsed_data.keys())] = list(parsed_data.values())
                idx_keep.append(i)
            except:
                idx_ignored.append(i)

        # Remove indices of tunes that weren't processed
        df = df.loc[idx_keep]
        print(f"Tunes processed. {len(df)} tunes were successfully processed...")

        # Transpose + convert to chroma
        df['tchroma'] = (df['abspitch'] - df['key']) % 12
        df['tchroma_octave'] = (df['abspitch'] - df['key'])
        
        # Update with melody length
        df['mel_len'] = df['tchroma'].apply(len)

        # Check for key changes
        pattern_key = r'\[K:[^\]]+\]'
        df['key_change'] = df['abc'].apply(lambda x: re.findall(pattern_key, x))
        df['has_key_change'] = df['key_change'].apply(len) > 0

        # Check for grace notes
        pattern_grace = r'\{[^}]+\}'
        df['has_grace'] = df['abc'].apply(lambda x: len(re.findall(pattern_grace, x)) > 0)

        # Check for polyphonic notes (within a single voice)
        pattern_poly = r'\[[^\]:]*\]'
        df['has_poly'] = df['abc'].apply(lambda x: len(re.findall(pattern_poly, x)) > 0)

        # Check for multiple voices
        pattern_voice = r'V:\d'
        df['has_voices'] = df['abc'].apply(lambda x: len(re.findall(pattern_voice, x)) > 0)

        # Check for consistency of repeat lines
        df['repeats_consistent'] = df['abc'].apply(TP.check_repeat_consistency)

        # Remove any tunes that music21 won't be able to parse properly
        if not full:
            df = df.loc[~(df.has_grace | df.has_poly | df.has_voices) & (df.repeats_consistent)]
            print(f"Tunes processed. {len(df)} tunes left after cleaning...")

        df.to_pickle(path)

        return df


### Primary extraction of melodic information, using music21
### Produces a dictionary, with setting_id as keys
def process_thesession_tunes_music21(df, json_data, redo=False):
    path = PATH_BASE.joinpath("Results", "thesession_music21.pkl")
    if path.exists() and not redo:
        return pickle.load(open(path, 'rb'))
    else:
        out = {}
        for i, setting in tqdm(zip(df.index, df.setting_id)):
            data = json_data[i]
            try:
                out[setting] = TP.parse_thesession_tune(data, alg='music21')
            except Exception as e:
                print(e)
        pickle.dump(out, open(path, 'wb'))
        return out


### Update the dictionaries with extra information.
### This requires information about the musical key
### that was obtained from the ABC header.
### It also performs a consistency check to make sure
### tunes were parsed correctly
def update_music21_data(df, tunes):
    setting_idx = {s:i for i, s in zip(df.index, df.setting_id)}
    keys2del = []
    for k, v in tunes.items():
        if k in setting_idx:
            for k2 in ['bar_onset', 'onsets', 'dur', 'midi']:
                v[k2] = np.array(v[k2], float)
            if v['onsets'].size != v['midi'].size:
                print('Size mismatch! ', k)
                keys2del.append(k)
            v['tmidi'] = v['midi'] - df.loc[setting_idx[k], 'key']
            v['tchroma'] = v['tmidi'] % 12
            tunes[k] = v
        else:
            keys2del.append(k)
    for k in keys2del:
        del tunes[k]
    return tunes


###############################################################
### Meertens data


### Meertens data is distributed across many files.
### This function finds all the file paths.
def load_meertens_paths():
    path_list = sorted(PATH_DATA.joinpath("Meertens", "MTC-FS-INST-2.0/krn").glob("*"))
    song_id, variant = np.array([p.stem.split('_') for p in path_list]).T
    return np.array([path_list, song_id, variant]).T


### Load tune type (vocal / instrumental) annotations
def load_meertens_metadata(df):
    cols = pd.read_csv(PATH_DATA.joinpath('Meertens/MTC-FS-INST-2.0/metadata/MTC-FS-INST-2.0-fieldnames.csv')).columns
    dfm = pd.read_csv(PATH_DATA.joinpath('Meertens/MTC-FS-INST-2.0/metadata/MTC-FS-INST-2.0.csv'), names=cols)
    key = {f:t for f, t in zip(dfm['filename'], dfm['type'])}
    df['type'] = df['ref'].map(key)
    return df



### Load meertens data:
### Creates a dataframe for metadata and summary stats,
### and a dictionary for the tune sequences
def load_meertens_data(redo=False):
    path_df = PATH_BASE.joinpath("Results", "meertens_summary.pkl")
    path_data = PATH_BASE.joinpath("Results", "meertens_tunes.pkl")
    if path_df.exists() and not redo:
        df = pd.read_pickle(path_df)
        data = pickle.load(open(path_data, 'rb'))
    else:
        df = pd.DataFrame(data=load_meertens_paths(), columns=['path', 'song_id', 'variant'])
        df['ref'] = [f"{a}_{b}" for a, b in zip(df.song_id, df.variant)]
        df = load_meertens_metadata(df)

        data = {}
        for i, p in tqdm(zip(df.index, df.path)):
            data[i] = TP.parse_kern_music21(p)

        df = update_meertens_df(df, data)

        df.to_pickle(path_df)
        pickle.dump(data, open(path_data, 'wb'))
    return df, data
    

### Updates the meertens dataframe
def update_meertens_df(df, data):
    df['ref'] = df.path.apply(lambda x: x.stem)
    df['key'] = [TP.get_key_idx(data[i]['key']) for i in df.index] 
    df['tchroma_octave'] = [(data[i]['midi'] - k) for i, k in zip(df.index, df.key)]
    df['tchroma'] = [(data[i]['midi'] - k) % 12 for i, k in zip(df.index, df.key)]
    return df



######################################################################
### Bronson


### Load Bronson data
def load_bronson_data(redo=False):
    path = PATH_DATA.joinpath('Bronson/merged_summary.pkl')
    df = pd.read_pickle(path)
    df['tmidi'] = df['midi'] - df['key']
    df = df.loc[df.tmidi.notnull()]
    df['tchroma_octave'] = [a.astype(int) for a in df['tmidi']]
    df['tchroma'] = [a.astype(int) for a in df['tmidi'] % 12]
    return df



