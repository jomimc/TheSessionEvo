from collections import defaultdict

from Bio import Seq, SeqRecord, SeqIO
import pandas as pd

from thesession.config import *
from thesession import utils

######################################################################
### Write to fasta


### Required format for writing fasta using SeqIO
def make_pitch_seqrecord(ID, seq, convert=True):
    if convert:
        seq = utils.tchroma2seq(seq)
    return SeqRecord.SeqRecord(Seq.Seq(seq), id=str(ID))


### Write fasta file, given SeqRecord (or list of SeqRecord objects)
def write_fasta(path, seqrecord):
    SeqIO.write(seqrecord, path, "fasta")


### Write many sequences to one fasta file
def write_all_seq_to_fasta(seq_list, id_list, path, convert=True):
    records = [make_pitch_seqrecord(ID, seq, convert) for ID, seq in zip(id_list, seq_list)]
    write_fasta(path, records)


### Write all thesession tunes to fasta
def write_all_seq_to_fasta_thesession(df):
    path = PATH_BASE.joinpath('fasta/all_seq_thesession.fasta')
    write_all_seq_to_fasta(df.tchroma, df.setting_id, path)


### Write all Meertens tunes to fasta
def write_all_seq_to_fasta_meertens(df):
    path = PATH_BASE.joinpath('fasta/all_seq_meertens.fasta')
    write_all_seq_to_fasta(df.tchroma, df.ref, path)


### Write all Bronson tunes to fasta
def write_all_seq_to_fasta_bronson(df):
    path = PATH_BASE.joinpath('fasta/all_seq_bronson.fasta')
    write_all_seq_to_fasta(df.tchroma, df.ref, path)


def write_fasta_tune_families_thesession(df, minsize=5):
    path_base = PATH_BASE.joinpath("Tools/mafft/thesession/tune_id/")
    tune_id_count = df.tune_id.value_counts()
    for t, c in tune_id_count.items():
        if c >= minsize:
            idx = df.tune_id == t
            records = [make_pitch_seqrecord(s, tp) for s, tp in zip(*df.loc[idx, ['setting_id', 'tchroma']].values.T)]
            path = path_base.joinpath(f"{t}/seq_list.fasta")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_fasta(path, records)


def write_parts_thesession(parts):
    path = PATH_BASE.joinpath('MMseqs/thesession_parts/all_seq_thesession_parts.fasta')
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [make_pitch_seqrecord(ID, part[0][1] % 12)  for ID, part in parts.items()]
    write_fasta(path, records)



######################################################################
### Load mmseqs results


def load_mmseqs_pairwise(df, dataset, annotate=True):
    path = PATH_BASE.joinpath(f'MMseqs/{dataset}/result.m8')
#   path = PATH_BASE.joinpath(f'MMseqs/tmp/result.m8')
    res = pd.read_csv(path, sep='\t')

    # Remove self-comparisons
    res = res.loc[res['query'] != res['target']]

    # Remove duplicates
    pairs = res[['query', 'target']].values
    df_sort = pd.DataFrame(np.sort(pairs, axis=1), columns=['a', 'b'])
    res = res.loc[df_sort.duplicated().values==False]

    if annotate:
        # Check whether hits are in the same tune family
        families, family_key = get_families_key(df, dataset)
        res = annotate_alignment(res, families, family_key)
    return res


def get_families_key(df, dataset):
    x, y = {'thesession_parts': ('tune_id', 'setting_id'),
            'thesession_tunes': ('tune_id', 'setting_id'),
            'meertens': ('song_id', 'ref'),
            'savage_english': ('chapter', 'ref')
            }[dataset]
    families = defaultdict(set)
    family_key = {}
    for t, s in zip(df[x], df[y]):
        families[t].add(s)
        family_key[s] = t
    return families, family_key


def annotate_alignment(res, families, family_key):
    res['in_fam'] = [t in families[family_key[q]] for q, t in zip(res["query"], res["target"])]
    return res


######################################################################
### Load mafft results


def load_msa(path):
    records = SeqIO.parse(path, 'fasta')
    return np.array([list(str(r.seq)) for r in records], str)


def load_tune_families_msa(dataset, df, minsize=5, alg='local'):
    path_base = PATH_BASE.joinpath(f"Tools/mafft/{dataset}/tune_id/")
    tune_id_count = df.tune_id.value_counts()
    msa_list = []
    for t, c in tune_id_count.items():
        if c >= minsize:
            path = path_base.joinpath(f"{t}/{alg}.align")
            msa_list.append(load_msa(path))
    return msa_list



