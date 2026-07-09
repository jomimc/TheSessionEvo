#!/usr/bin/env bash
# Build the two Zenodo zips from the working tree into ./Zenodo/.
# Author-run on Linux; end users only need scripts/download_data.py.
#
# The include lists below were determined empirically (see
# docs/code_release_audit.md) by running each data_for_figN / make_figures
# step with redo=False and checking which cache files are actually touched.
# Notably, Cache/ParameterOptimizationSearch/ and all four MMseqs/*/result.m8
# files ARE required (there is no higher-level cache that bypasses them), so
# they are NOT excluded despite being the largest items.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
OUT=Zenodo
rm -rf "$OUT"
mkdir -p "$OUT"

# Tier 1: figure data — enough to run make_figures() end to end.
# Three spots read raw inputs directly rather than going through
# FigureData/, each discovered by an actual fresh-clone smoke test (copy
# the code only, populate from this zip alone, call make_figures()):
#   main_figs.fig2() reads ProteinData/{accession}/*.csv,*.npy (~1.5 MB;
#     afdb_cache/, 14 MB of AlphaFold PDBs, is Tier-2-only — only needed to
#     regenerate, not to render from existing outputs).
#   main_figs.fig3() reads Data/SavageFig/{English,Japanese}.txt (8 KB).
#   si_figs.plot_optimization_scores() reads
#     Cache/ParameterOptimizationSearch/ (the biggest of the three, but it
#     compresses well — see the du -sh output this script prints).
zip -r "$OUT/tier1_figuredata.zip" FigureData -x '*.DS_Store'
for acc in P00004 P0AA25 P0AE67; do
    zip -r "$OUT/tier1_figuredata.zip" "ProteinData/$acc" -x '*.fasta' -x '*afdb_cache*'
done
zip -r "$OUT/tier1_figuredata.zip" Data/SavageFig -x '*.png'
zip -r "$OUT/tier1_figuredata.zip" Cache/ParameterOptimizationSearch

# Tier 2: caches + alignment outputs + protein data + non-fetchable inputs.
# Excludes rerun-only intermediates that no redo=False path reads:
#   Cache/thesession_full.pkl, Cache/thesession_clean.pkl (superseded by
#     thesession_cleaned_processed.pkl / thesession_music21.pkl)
#   MMseqs/tmp/ (stray leftover from an interrupted run, not part of any
#     dataset's output)
zip -r "$OUT/tier2_caches.zip" \
    Cache/thesession_cleaned_processed.pkl \
    Cache/thesession_music21.pkl \
    Cache/thesession_tunes.pkl \
    Cache/meertens_summary.pkl \
    Cache/meertens_tunes.pkl \
    Cache/savage_full.pkl \
    Cache/all_parts_thesession.pkl \
    Cache/all_parts_thesession_df.pkl \
    Cache/pairs_thesession_parts.pkl \
    Cache/pairs_thesession_parts_hits.pkl \
    Cache/pairs_thesession_parts_mismatches.pkl \
    Cache/ParameterOptimizationSearch \
    MMseqs/thesession_parts \
    MMseqs/thesession_tunes \
    MMseqs/meertens \
    MMseqs/savage_english \
    MMseqs/substitution_matrices \
    ProteinData \
    Data/SavageFig

# Checksums + lockfile
( cd "$OUT" && sha256sum ./*.zip > MANIFEST.sha256 )
pip freeze > "$OUT/requirements-lock.txt"
cp README_DATA.md "$OUT/README_DATA.md"
echo "Wrote $OUT/{tier1_figuredata.zip,tier2_caches.zip,MANIFEST.sha256,requirements-lock.txt,README_DATA.md}"
du -sh "$OUT"/*.zip
