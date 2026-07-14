# Zenodo data archive — what's in it, and how to use it

This file ships in the Zenodo record alongside the two data zips. It explains
what each tier contains, how to unpack it, and where the frozen upstream
commits come from.

## Tiers

- **Tier 1 — `tier1_figuredata.zip` (~1 MB).** Unpack at the repo root →
  `FigureData/`, plus `ProteinData/{P00004,P0AA25,P0AE67}/` and
  `Data/SavageFig/`. Everything `make_figures()` reads.
- **Tier 2 — `tier2_caches.zip` (~2.5 GB).** Unpack at the repo root →
  `Cache/`, `MMseqs/`, `ProteinData/`, `Data/SavageFig/`. Adds the parsed
  corpus caches, MMseqs2 alignment tables, and protein outputs, so any
  `data_for_figN` step  in the pipeline can be re-run (with a different parameter, say)
  **without** installing MMseqs2, BLAST+, or MAFFT, and without re-parsing
  the raw corpora. This tier also *is* the version pin: the cached `.pkl`
  files fix the exact TheSession snapshot the paper used.


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

## Checksums

`MANIFEST.sha256` (in this same Zenodo record) lists the SHA-256 of both
zips. Verify after download:

```bash
sha256sum -c MANIFEST.sha256
```
