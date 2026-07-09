# TheSessionEvo

**Inferring principles of cultural evolution through rhythm-aware alignment of 40,000 Irish folk tunes**

This project adapts bioinformatics sequence-alignment methods to study the large-scale
evolution of Irish folk melodies. Melodies are encoded as one-dimensional sequences and
aligned using a custom rhythm-aware algorithm, enabling statistical analysis of evolutionary
forces in musical transmission.

## Overview

We treat melodic variants of the same tune as analogues to homologous protein sequences,
applying tools from molecular evolution (substitution matrices, covariance analysis,
conservation scoring) to music. The key methodological contribution is a rhythm-aware
alignment algorithm that respects metrical structure — something standard bioinformatics
tools cannot handle.

**Core findings:**
- Pitch mutability correlates with tonal hierarchy
- Substitution rates mirror interval distributions
- Stronger metrical positions are more conserved
- Repetition structure is conserved across variants

## Requirements

| Dependency | Kind | Used for | Optional? |
|---|---|---|---|
| Python 3.12+ | runtime | everything | required |
| numpy, pandas, scipy | py | all numerical work | required |
| music21 | py | parse ABC/kern, pitch extraction | required to parse raw corpora; **not** needed if using Tier-2 caches |
| biopython | py | FASTA/alignment IO, protein fetch | required |
| scikit-learn, statsmodels | py | ROC/AUC, regressions, bootstrap | required |
| matplotlib, seaborn | py | figures | required for figures |
| networkx | py | substitution-matrix graph (SI3) | required for that figure |
| parasail | py | BLOSUM/submat alignment in viz | required for fig2/SI panels |
| openpyxl | py | read Savage `.xlsx` | required only for the Savage leg |
| tqdm | py | progress bars | required (cosmetic; cheap to keep) |
| requests | py | UniProt/NCBI/AlphaFold fetch | **optional** — only to regenerate protein data from scratch |
| **MMseqs2** (`mmseqs`) | binary | all-vs-all melodic alignment | **optional** — only to re-run alignment from raw; Tier-2 ships `result.m8` |
| **NCBI BLAST+** (`blastp`) + a protein DB | binary | protein homolog search | **optional** — only to regenerate protein data; Tier-2 ships outputs |
| **MAFFT** (`mafft`) | binary | protein MSA | **optional** — same as BLAST+ |

Install the Python dependencies with `pip install -e .` from the repo root (see `pyproject.toml`).

## Quickstart — reproduce a figure in a few minutes

```bash
pip install -e .
python scripts/download_data.py --tier figures
python -c "import run_pipeline; run_pipeline.make_figures()"
```
No MMseqs2, no raw parsing — this uses the cached `FigureData/` shipped in Zenodo Tier 1.
The same three commands work unchanged on Windows, macOS, and Linux.

## Usage

**Interactive analysis:**
```bash
python Src/start_ipython.py
```
This pre-loads all modules and common data objects for interactive exploration.

**Full pipeline:**
```bash
python Src/run_pipeline.py
```
The pipeline runs end-to-end: parsing ABC notation → extracting tune parts → running
MMseqs2 all-vs-all alignment → statistical analysis → figure generation. Individual
steps live in `thesession.pipeline` (`fig1`..`fig5`, `mmseqs`, `param_search`, `protein`)
and can be re-run in isolation with `redo=True` to recompute just that step.

**Re-run an analysis with different parameters** (no MMseqs2/BLAST/MAFFT needed):
```bash
python scripts/download_data.py --tier full
# edit the relevant thesession/pipeline/figN.py step, then run it directly, e.g.:
python -c "from thesession.pipeline.fig3 import data_for_fig3; ..."
```

**Reproduce from raw data** (needs MMseqs2, and BLAST+/MAFFT for the protein leg):
```bash
python scripts/fetch_data.py     # TheSession @4f2d9d5, Savage .xlsx @82a8625
# Meertens requires a signed form — see DATA.md
python Src/run_pipeline.py
```

Results are cached in `Cache/`; set `redo=True` in individual steps to recompute.
Generated figures are saved to `Figures/`.

## Project Structure

```
Src/
├── run_pipeline.py           # Thin driver: main() + make_figures()
├── start_ipython.py          # Interactive session setup
└── thesession/
    ├── config.py             # Global constants, paths, musical mode definitions
    ├── utils.py              # Sequence conversion, statistics utilities
    ├── io/                   # Data loading: ABC notation, FASTA, DataFrames
    ├── alignment/             # Pairwise and rhythm-aware part alignment
    ├── structure/             # Tune part separation (A/B sections)
    ├── analysis/              # Substitution matrices, key/mode analysis, optimisation
    ├── protein/               # Protein conservation/covariance/structure pipeline
    ├── pipeline/               # Figure-data generation steps (fig1..fig5, mmseqs, protein)
    └── viz/                   # Manuscript figure generation
scripts/                      # fetch_data.py, download_data.py, build_zenodo.sh
Data/                          # Raw input corpora (fetched or symlinked; see Data section)
Cache/                        # Cached intermediate data (pkl, npy)
FigureData/                   # Numerical data backing each figure
Figures/                      # Generated PDF/PNG figures
MMseqs/                       # MMseqs2 FASTA inputs and result tables
ProteinData/                  # Protein-comparison outputs (conservation, RSA, contacts)
```

## Data

Input data are drawn from three sources:

- **TheSession** — Irish/Celtic tunes in ABC notation.
  [github.com/adactio/TheSession-data](https://github.com/adactio/TheSession-data)
  — fetchable, pinned to commit `4f2d9d5` (2026-03-15).
- **Meertens Tune Collections** — Dutch folk music collection.
  [liederenbank.nl/mtc](https://www.liederenbank.nl/mtc) — form-gated, not
  redistributable; see `DATA.md` for the manual download.
- **Savage et al.** — Manually-aligned melody pairs from the Bronson ballad collection.
  [github.com/pesavage/melodic-evolution](https://github.com/pesavage/melodic-evolution/tree/master/data)
  — fetchable, pinned to commit `82a8625` (2021-11-19).

Raw data paths default to `<repo>/Data`, configured in `Src/thesession/config.py`;
override with the `THESESSION_DATA` environment variable to use an out-of-tree
data directory. Use `python scripts/fetch_data.py` to populate `Data/` for the two
fetchable sources.

A stable version of the data and code is archived on Zenodo. \[LINK\] The archive is
split into two tiers — see `README_DATA.md` for what each contains and how to fetch
them (`scripts/download_data.py`).

