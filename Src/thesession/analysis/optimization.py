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


### Run mmseqs search with inputs:
###     fasta (all sequences, for all-vs-all comparison)
###     submat (substitution matrix, with match/mismatch scores)
###     gap_open, gap_extend (gap penalties, given as positive numbers)
### Outputs are saved in path_result, with temporary files in path_tmp
def run_mmseqs(path_fasta, path_result, path_tmp, path_submat, gap_open, gap_extend):
    args = [MMSEQS_BIN, 'easy-search', str(path_fasta), str(path_fasta),
            str(path_result), str(path_tmp), '--format-mode', '4',
            '--sub-mat', str(path_submat), '--gap-open', str(gap_open),
            '--gap-extend', str(gap_extend)]

    pipe_output = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = pipe_output.communicate()
    return stdout, stderr



def explore_parameter_space(df, dataset='thesession_tunes'):
    # Set up paths to input / outputs
    path_base = PATH_MMSEQS.joinpath(f'{dataset}')
    path_fasta = path_base.joinpath(f"all_seq_{dataset}.fasta")
    path_mmseqs = PATH_MMSEQS.joinpath("tmp", "result.m8")

    path_submat_list = sorted(PATH_MMSEQS.joinpath("substitution_matrices").glob("*.out"))
    path_results = PATH_CACHE.joinpath("ParameterOptimizationSearch", dataset)
    path_results.mkdir(parents=True, exist_ok=True)

    # Load tune family annotations
    families, family_key = seq_io.get_families_key(df, dataset)
    xkey = {'thesession_tunes':'tune_id',
            'meertens':'song_id',
            'savage_english':'chapter'
            }[dataset]

    gap_open = np.arange(2, 16, 1)
    data = []
    for path_submat in path_submat_list:
#       if path_submat.stem[0] != 'A':
#           continue
        for go in gap_open:
            gap_extend = np.arange(1, go + 1, 1)
            for ge in gap_extend:
                path = path_results.joinpath(f"{path_submat.stem}_{go}_{ge}.json")
                print(path)
                if path.exists():
                    try:
                        data.append(json.load(open(path, 'r')))
                    except:
                        os.remove(path)

                else:
                    print(path)
                    stdout, stderr = run_mmseqs(path_fasta, path_mmseqs, path_mmseqs.with_name("tmp"),
                                                path_submat, go, ge)
                    print(stdout)
                    print(stderr)

                    out = {'path_submat':str(path_submat), 'gap_open':float(go), 'gap_extend':float(ge)}

                    if len(stderr) == 0:
                        res = pd.read_csv(path_mmseqs, sep='\t')
                        res = seq_io.annotate_alignment(res, families, family_key)
                        out = get_roc_and_auc(res, dataset, out)
                        out = get_total_positives(df, dataset, xkey, out)
                        out['auc'] = float(roc_auc_score(res.in_fam, res.fident))

                        tpr, fpr = calculate_actual_rates(out)
                        out['actual_tpr'] = tpr
                        out['actual_fpr'] = fpr
                        out['mean_precision'] = calculate_average_precision(out)

                    data.append(out)
                    json.dump(out, open(path, 'w'))
                    shutil.rmtree(path_mmseqs.with_name("tmp"))
#               return


def get_total_positives(df, dataset, x='tune_id', fig_data={}):
    total = len(df)**2
    positives = np.sum([n * (n - 1) / 2 for n in df[x].value_counts().values])
    negatives = total - positives
    fig_data[f'positives'] = int(positives)
    fig_data[f'negatives'] = int(negatives)
    fig_data[f'total'] = int(total)
    return fig_data 


# Get roc and roc-auc
def get_roc_and_auc(res, dataset, fig_data={}):
    fpr, tpr, _ = roc_curve(res.in_fam, res.fident)
    auc = roc_auc_score(res.in_fam, res.fident)

    # Save to container
    fig_data[f'fpr'] = list(fpr)
    fig_data[f'tpr'] = list(tpr)
    fig_data[f'auc'] = auc
    fig_data[f'screened'] = len(res)
    fig_data[f'screened_positives'] = int(np.sum(res.in_fam))
    fig_data[f'screened_negatives'] = int(len(res) - np.sum(res.in_fam))
    return fig_data


def calculate_actual_rates(data):
    tpr = data[f"screened_positives"] / data[f"positives"]
    fpr = data[f"screened_negatives"] / data[f"negatives"]
    return tpr, fpr


def get_precision_recall(data):
    names = ['total', 'positives', 'negatives', 'screened_positives',
             'screened_negatives']
    N, Nt, Nf, Mt, Mf = [data[x] for x in names]
    fpr, tpr = [np.array(data[x]) for x in ['fpr', 'tpr']]
    precision = (tpr * Mt) / (tpr * Mt + fpr * Mf)
    recall = (tpr * Mt) / Nt
    return precision, recall


def calculate_average_precision(data):
    precision, recall = get_precision_recall(data)
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])



def parse_filename(filename):
    mat_type, diag, off_diag, gap_open, gap_extend = filename.split('_')
    return mat_type, diag, off_diag, gap_open, gap_extend


def load_results_mmseqs(dataset):
    path_results = PATH_CACHE.joinpath("ParameterOptimizationSearch", dataset)
    path_list = sorted(path_results.glob("*json"))
    data = []
    for path in path_list:
        mat_type, diag, off_diag, gap_open, gap_extend = parse_filename(path.stem)
#       if mat_type != 'A':
#           continue
        out = json.load(open(path, 'r'))
        out['mat_type'] = mat_type
        out['diag'] = diag
        out['off_diag'] = off_diag
        data.append(out)
    return pd.DataFrame(data)


########################################################################################
### Optimizing parameters for NW pairwise alignments


def optimize_alignment_savage():
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
    match = np.arange(2, 12, 2)
    mismatch = np.arange(-4, 1, 1)
    gap_open = np.arange(2, 16, 1)
    params = []
    for ma in match:
        for mi in mismatch:
            for go in gap_open:
                gap_extend = np.arange(1, go + 1, 1)
                for ge in gap_extend:
                    params.append((ma, mi, go, ge))
    return params
                    

def load_results_savage():
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
        overall.append(correct)
    dfr = pd.DataFrame(data=output)
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



