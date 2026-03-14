import music21

from global_variables import *
import load_tunes, seq_io, plots, pyabc, utils, seq_align, run_main, savage, main_figs, si_figs
import substitution_matrix as SM
import onset_align as OA
import optimize_parameters as OP
import tune_parser as TP
import part_alignments as PA
import part_separation as PS

df, tunes = load_tunes.load_thesession_data(redo=False)
df_parts, parts_data = PS.get_all_parts_thesession(df, tunes, redo=False)
#res = run_main.load_mmseqs_parts(parts_data, "thesession_parts", redo=False, annotate=False)
res = seq_io.load_mmseqs_pairwise(df, "thesession_parts", annotate=False)
res = PA.prune_identical_parts(res, parts_data)
res, res0, mismatches = PA.annotate_res(df, df_parts, res, parts_data, redo=False)



