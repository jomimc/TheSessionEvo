import music21

from thesession.config import *
from thesession.io import tune_loader as load_tunes, seq_io, pyabc
from thesession.alignment import pairwise as seq_align
from thesession.io import savage_loader as savage
from thesession.viz import main_figs, si_figs
from thesession import utils
import plots
import run_pipeline as run_main
from thesession.analysis import substitution as SM
from thesession.alignment import onset as OA
from thesession.analysis import optimization as OP
from thesession.io import tune_parser as TP
from thesession.alignment import parts as PA
from thesession.structure import part_separation as PS

df, tunes = load_tunes.load_thesession_data(redo=False)
df_parts, parts_data = PS.get_all_parts_thesession(df, tunes, redo=False)
#res = run_main.load_mmseqs_parts(parts_data, "thesession_parts", redo=False, annotate=False)
res = seq_io.load_mmseqs_pairwise(df, "thesession_parts", annotate=False)
res = PA.prune_identical_parts(res, parts_data)
res, res0, mismatches = PA.annotate_res(df, df_parts, res, parts_data, redo=False)



