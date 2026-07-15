#!/usr/bin/env bash
# Build the two Zenodo zips from the working tree into ./Zenodo/.
 
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
OUT=Zenodo
rm -rf "$OUT"
mkdir -p "$OUT"

# Tier 1: figure data — enough to run make_figures() end to end.

zip -r "$OUT/tier1_figuredata.zip" FigureData -x '*.DS_Store'
for acc in P00004 P0AA25 P0AE67; do
    zip -r "$OUT/tier1_figuredata.zip" "ProteinData/$acc" -x '*.fasta' -x '*afdb_cache*'
done
zip -r "$OUT/tier1_figuredata.zip" Data/SavageFig -x '*.png'
zip -r "$OUT/tier1_figuredata.zip" Cache/ParameterOptimizationSearch

# Tier 2: caches + alignment outputs + protein data + non-fetchable inputs.
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
    MMseqs/substitution_matrices \
    ProteinData \
    Data/SavageFig

# Checksums + lockfile
( cd "$OUT" && sha256sum ./*.zip > MANIFEST.sha256 )
cp README_DATA.md "$OUT/README_DATA.md"
echo "Wrote $OUT/{tier1_figuredata.zip,tier2_caches.zip,MANIFEST.sha256,requirements-lock.txt,README_DATA.md}"
du -sh "$OUT"/*.zip
