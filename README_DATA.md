# Zenodo data archive — what's in it, and how to use it

This file ships in the Zenodo record alongside the two data zips. It explains
what each tier contains, how to unpack it, and where the frozen upstream
commits come from.

## Tiers

- **Tier 1 — `tier1_figuredata.zip` (~1 MB).** Unpack at the repo root →
  `FigureData/`, plus `ProteinData/{P00004,P0AA25,P0AE67}/` and
  `Data/SavageFig/`. Everything `make_figures()` reads — this was confirmed
  with an actual fresh-clone smoke test (copy of the code, no data
  directories, populate only from this zip, run `make_figures()`), which is
  how the latter two additions were caught: `main_figs.fig2()` and `fig3()`
  read `ProteinData/` and `Data/SavageFig/` directly rather than through
  `FigureData/`. No MMseqs2, no parsing, seconds to reproduce or tweak a
  figure.
- **Tier 2 — `tier2_caches.zip` (~2.5 GB).** Unpack at the repo root →
  `Cache/`, `MMseqs/`, `ProteinData/`, `Data/SavageFig/`. Adds the parsed
  corpus caches, MMseqs2 alignment tables, and protein outputs, so any
  `data_for_figN` step can be re-run (with a different parameter, say)
  **without** installing MMseqs2, BLAST+, or MAFFT, and without re-parsing
  the raw corpora. This tier also *is* the version pin: the cached `.pkl`
  files fix the exact TheSession snapshot the paper used.

  Tier 2 is larger than earlier estimates because two things turned out to
  be required, not prunable: `Cache/ParameterOptimizationSearch/` (read
  directly by the SI "optimization scores" figure) and all four
  `MMseqs/*/result.m8` tables (there is no higher-level cache that lets
  `data_for_fig1` or `run_main_alignments` skip reading them). This was
  verified empirically by hiding each file and re-running the relevant step.

  `Cache/thesession_full.pkl` and `Cache/thesession_clean.pkl` are excluded
  — they're intermediates from raw parsing that no `redo=False` path reads;
  they're only touched when reprocessing from Tier 3 raw inputs.

Unpacking either zip at the repo root lands every file at the path
`Src/thesession/config.py`'s `PATH_*` constants expect — no code changes
needed. Use `scripts/download_data.py` to fetch and unpack automatically.

## Non-fetchable inputs bundled in Tier 1 and Tier 2

- `Data/SavageFig/English.txt` and `Japanese.txt` — digitized from Fig. 4B of
  Savage et al.'s paper. They exist in no public repository, so they can't be
  fetched by `scripts/fetch_data.py`; they ship in both tiers (tiny, ~8 KB).

## Frozen upstream commits (Tier 3 — see `scripts/fetch_data.py`)

- `adactio/TheSession-data` → `4f2d9d551f941caacb9d1f3a37721b319108bce7`
  (2026-03-15).
- `pesavage/melodic-evolution` (Savage) →
  `82a8625fae4ecc5ee3141d6e48bf3ae3b554d59b` (2021-11-19).

A discrepancy was flagged during the code-release audit and has since been
resolved by the owner: `data/MelodicEvoSeq.xlsx` initially differed in size
between the pinned-commit fetch (~526 KB) and the file previously archived
here (~325 KB), even though its sibling `MelodicEvoSeqFullSongs.xlsx`
matched byte-for-byte at the same commit. The owner inspected both and
confirmed the content is equivalent; the archive now uses the GitHub
(pinned-commit) version, matching what `scripts/fetch_data.py` produces.

## Checksums

`MANIFEST.sha256` (in this same Zenodo record) lists the SHA-256 of both
zips. Verify after download:

```bash
sha256sum -c MANIFEST.sha256
```
