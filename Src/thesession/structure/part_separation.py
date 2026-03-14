"""
Algorithm for separating tunes into parts including evaluationg code.


"""
from collections import Counter, defaultdict
import json
import pickle
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesession.io import tune_loader as load_tunes
from thesession.io import tune_parser as TP
from thesession.config import *
from thesession import utils


#######################################################
### Load bars

### Separate a setting into its bars
def extract_bars(tunes):
    bars, dur, tmidi = [tunes[x] for x in ['bars', 'dur', 'tmidi']]
    bar_dur, bar_tmidi = [], []
    for i in range(len(bars) - 1):
        bar_dur.append(dur[bars[i]:bars[i+1]])
        bar_tmidi.append(tmidi[bars[i]:bars[i+1]])

    bar_dur.append(dur[bars[-1]:])
    bar_tmidi.append(tmidi[bars[-1]:])

    bar_total_dur = [np.sum(d) for d in bar_dur]

    return np.array(bar_total_dur), bar_dur, bar_tmidi


### Remove pickups and check that durations add up
def screen_bars(tunes, bar_len=None, min_bars=8):
    bar_total_dur, bar_dur, bar_tmidi = extract_bars(tunes)

    if isinstance(bar_len, type(None)):
        bar_len = np.max(bar_total_dur)

    # Find bars that are not long enough
    idx = np.where(np.array(bar_total_dur) < bar_len)[0]

    # Check for bars with missing beats (apart from pickup)
    if np.sum(idx != 0) > 0:
        return False, [], []

    # Check for pickup
    if 0 in idx:
        bar_tmidi = bar_tmidi[1:]
        bar_dur = bar_dur[1:]

    # Check for correct length of tune
    if (len(bar_tmidi) % min_bars) != 0:
        return False, [], []

    return True, bar_dur, bar_tmidi


#######################################################
### Separate tunes into parts


### This algorithm starts by separating parts into 8-bar segments,
### and then merges neighboring segments if they are very similar.
### The output is a list of parts.
### For most tunes, min_bars = 8. For slides, this should techically be 4.
def separate_tune_into_parts(tmidi, dur, cutoff=0.8, min_bars=8):
    # Join bars into parts composed of "min_bars" bars each
    nparts = len(tmidi) // 8
    parts_init = [(np.concatenate(dur[i*8:(i+1)*8]), 
                   np.concatenate(tmidi[i*8:(i+1)*8])) for i in range(nparts)]

    # Calculate similarity between neighboring parts
    part_similarity = np.zeros(nparts - 1)
    for i in range(nparts - 1):
        factor = utils.get_common_denominator([parts_init[i][0], parts_init[i+1][0]])
        if factor == 0:
            print(set(np.concatenate([parts_init[i][0], parts_init[i+1][0]])))
            raise Exception("Common denominator not found!")

        tc_grid1 = utils.get_tchroma_grid(parts_init[i][1] % 12, parts_init[i][0], factor)
        tc_grid2 = utils.get_tchroma_grid(parts_init[i+1][1] % 12, parts_init[i+1][0], factor)
        part_similarity[i] = np.mean(tc_grid1 == tc_grid2)

    # If no neighboring parts are similar, return the initial set of parts
    if np.all(part_similarity < cutoff):
        return parts_init, np.ones(nparts) * 8

    # Merge parts in order of appearance. Parts can only be merged once.
    i = 0
    parts_final = []
    part_nbars = []
    while i < (nparts):
        # For the last part, we check if it was merged, and if not we add it
        if i == nparts - 1:
            if part_similarity[i-1] < cutoff:
                parts_final.append(parts_init[i])
                part_nbars.append(8)
                i += 1
            else:
                i += 1
        else:
            if part_similarity[i] >= cutoff:
                parts_final.append((np.concatenate([parts_init[i][0], parts_init[i+1][0]]),
                                    np.concatenate([parts_init[i][1], parts_init[i+1][1]])))
                part_nbars.append(16)
                i += 2
            else:
                parts_final.append(parts_init[i])
                part_nbars.append(8)
                i += 1

    return parts_final, np.array(part_nbars)


### Given a tune_id, load all settings and split them into parts
def get_all_parts_tune(df, tunes, tune_id, min_bars=8):
    all_parts = []
    to_keep = []
    for k, setting_id in enumerate(df.loc[df.tune_id==tune_id, 'setting_id']):
        # Remove bars that do not add up to the expected duration for the given meter.
        # This does two things:
        #   Removes pickups
        #   Removes any bars with incorrect durations (rare, but it happens)
        # If any bars with incorrect durations are removed, then the total number of bars
        # will not be a multiple of 8, and it will be discarded
        use_okay, bar_dur, bar_tmidi = screen_bars(tunes[setting_id], min_bars=min_bars)

        if use_okay:
            all_parts.append(separate_tune_into_parts(bar_tmidi, bar_dur, min_bars=min_bars))
            to_keep.append(k)
    return all_parts, to_keep


### Load all parts from TheSession tunes
### (code takes about 2 hours to run)
def get_all_parts_thesession(df, tunes, redo=False):
    path_df = PATH_CACHE.joinpath("all_parts_thesession_df.pkl")
    path = PATH_CACHE.joinpath("all_parts_thesession.pkl")
    if all([path_df.exists(), path.exists()]) and not redo:
        return pickle.load(open(path, 'rb')), pd.read_pickle(path_df)
    else:
        cols = ["part_id", "tune_id", "setting_id", "part_no", 'num_parts']
        rows = []
        parts_out = {}
        ts = time()
        for tune_id in tqdm(df.tune_id.unique()):
#       for tune_id in df.tune_id.unique():
            settings = df.loc[df.tune_id==tune_id, 'setting_id'].values
            all_parts, to_keep = get_all_parts_tune(df, tunes, tune_id)
            for j, parts in zip(to_keep, all_parts):
                for k, (part, nbars) in enumerate(zip(*parts)):
                    part_id = f"{tune_id}_{settings[j]}_{k}"
                    # Data structure of "p": (parts (dur, tmidi), nbars_per_part)
                    parts_out[part_id] = (part, nbars)
                    rows.append([part_id, tune_id, settings[j], k, len(to_keep)])
        print("Time taken: ", (time() - ts) / 60)
        pickle.dump(parts_out, open(path, 'wb'))
        df_parts = pd.DataFrame(data=rows, columns=cols)
        return df_parts, parts_out


#ef update_parts_df(df_parts):



#######################################################
### Evaluate part separation algorithm


### Create a template json file for annotating ground truth
def save_gt_part_template(df, tune_id):
    path = PATH_DATA.joinpath("TheSession-data/part_annotations", f'tune_{tune_id}.json')
    if path.exists():
        print("File already exists!")
    else:
        setting_id_list = df.loc[df.tune_id==tune_id, 'setting_id'].values
        out = [{"setting_id":int(ID), "part_start":{}} for ID in setting_id_list]
        with open(path, 'w') as o:
            o.write('},\n'.join(json.dumps(out).split('},')))


### Load annotated ground truth for number of parts in settings
def load_gt_parts(tune_id):
    path = PATH_DATA.joinpath("TheSession-data/part_annotations", f'tune_{tune_id}.json')
    gt = json.load(open(path, 'r'))
    return gt


### Evaluate part separation algorithm for a tune family
def evaluate_nparts(df, tunes, tune_id, cutoff=0.8, verbose=False, min_bars=8):
    setting_list = df.loc[df.tune_id==tune_id, 'setting_id'].values

    # Load ground-truth
    gt = load_gt_parts(tune_id)
    gt_settings = np.array([x['setting_id'] for x in gt if x['setting_id'] in setting_list])
    gt_nparts = np.array([len(x['part_start']) for x in gt if x['setting_id'] in setting_list])

    # Calculate the number of parts per tune
    nparts = []
    for s in setting_list:
        if s not in gt_settings:
            continue
        use_okay, bar_dur, bar_tmidi = screen_bars(tunes[s], min_bars=min_bars)
        if use_okay:
            nparts.append(len(separate_tune_into_parts(bar_tmidi, bar_dur, cutoff)[0]))
        else:
            nparts.append(0)
    nparts = np.array(nparts)

    # Some tunes have errors. Don't include these in the accuracy calculation
    idx = nparts != 0
    acc = np.mean(nparts[idx] == gt_nparts[idx])
    if verbose:
        print(nparts)
        print(gt_nparts)
        print(f"Accuracy = {acc}")
        print(f"Incorrect parts:")
        for i in range(len(nparts)):
            if nparts[i] != gt_nparts[i]:
                print("\t", i, nparts[i], gt_nparts[i], len(dm[to_keep[i]]['bars']) / nparts[i])
    return acc


### Evaluate part separation algorithm across all tune families
def evaluate_nparts_params(df, tunes):
    tune_list = [2, 21, 27, 34, 62, 74]
    cutoff_list = np.arange(0.5, 1, 0.05)
    acc = []
    for t in tune_list:
        for cutoff in cutoff_list:
            acc.append(evaluate_nparts(df, tunes, t, cutoff=cutoff))
    return np.array(acc).reshape(len(tune_list), len(cutoff_list))




