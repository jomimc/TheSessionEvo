# TheSessionEvo

**Inferring principles of cultural evolution through rhythm-aware alignment of 40,000 Irish folk tunes**

This project adapts bioinformatics sequence-alignment methods to study the large-scale
evolution of Irish folk melodies: melodies are encoded as one-dimensional sequences and
aligned with a custom rhythm-aware algorithm, enabling statistical analysis of evolutionary
forces in musical transmission.

📄 **For the motivation, methods, and findings, see the paper:** _[title / DOI — TODO]_.
This README covers only how to install the code and reproduce the analyses and figures.

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
No MMseqs2, no raw parsing — this uses the cached `FigureData/` shipped in the data archive
(Tier 1). The same three commands work unchanged on Windows, macOS, and Linux.

## Notebooks

The `notebooks/` directory has a guided, runnable notebook for each data tier:

| Notebook | Needs | What it does |
|---|---|---|
| `01_tier1_reproduce_figures.ipynb` | Tier 1 (~48 MB) | Re-render every published figure from `FigureData/`. Seconds; no MMseqs2. |
| `02_tier2_rerun_analysis.ipynb` | Tier 2 (~483 MB) | Load the cached corpus and re-run any analysis without MMseqs2 or re-parsing. |
| `03_tier3_full_pipeline.ipynb` | raw data + binaries | Full reproduction from scratch (needs MMseqs2, and BLAST+/MAFFT for the protein leg). |

**Locally:**
```bash
pip install -e .
jupyter lab notebooks/        # or: jupyter notebook notebooks/
```
Open a notebook and run the cells top to bottom. Each one downloads the data tier it needs
(or tells you how) and adds `Src/` to the path, so it works with or without `pip install -e .`.

**In Google Colab:** open a notebook directly from GitHub and run it — its first cell
bootstraps the environment (clones this repo, installs it, downloads the tier's data) when it
detects Colab, and does nothing when run inside a local clone.

- Tier 1: [open in Colab](https://colab.research.google.com/github/jomimc/TheSessionEvo/blob/main/notebooks/01_tier1_reproduce_figures.ipynb)
- Tier 2: [open in Colab](https://colab.research.google.com/github/jomimc/TheSessionEvo/blob/main/notebooks/02_tier2_rerun_analysis.ipynb)
- Tier 3: [open in Colab](https://colab.research.google.com/github/jomimc/TheSessionEvo/blob/main/notebooks/03_tier3_full_pipeline.ipynb)

> Data download needs the archive URLs set in `scripts/download_data.py` (pending the published
> record). Tier 3 additionally needs MMseqs2 (+ BLAST+/MAFFT); the notebook shows how to install
> them on Colab.

## Usage

**Interactive analysis:**
```bash
python Src/start_ipython.py
```
Pre-loads all modules and the common data objects for interactive exploration.

**Full pipeline:**
```bash
python Src/run_pipeline.py
```
Runs end-to-end: parsing ABC notation → extracting tune parts → running MMseqs2 all-vs-all
alignment → statistical analysis → figure generation. The individual analysis stages live in
`thesession.pipeline` (`identification`, `mutability`, `substitution`, `position`, `covariance`,
plus `mmseqs`, `param_search`, `protein`) and can be re-run in isolation with `redo=True`.

**Re-run an analysis with different parameters** (no MMseqs2/BLAST/MAFFT needed):
```bash
python scripts/download_data.py --tier full
# edit the relevant thesession/pipeline/<stage>.py step, then run it directly, e.g.:
python -c "from thesession.pipeline.substitution import data_for_substitution; ..."
```

**Reproduce from raw data** (needs MMseqs2, and BLAST+/MAFFT for the protein leg):
```bash
python scripts/fetch_data.py     # TheSession @4f2d9d5, Savage .xlsx @82a8625
# Meertens requires a signed form — see DATA.md
python Src/run_pipeline.py
```

Results are cached in `Cache/`; set `redo=True` in individual steps to recompute.
Generated figures are saved to `Figures/`.

## Project structure

```
Src/
├── run_pipeline.py           # Thin driver: main() + make_figures()
├── start_ipython.py          # Interactive session setup
└── thesession/
    ├── config.py             # Global constants, paths, musical mode definitions
    ├── utils.py              # Sequence conversion, statistics utilities
    ├── io/                   # Data loading: ABC notation, FASTA, DataFrames
    ├── alignment/            # Pairwise and rhythm-aware part alignment
    ├── structure/            # Tune part separation (A/B sections)
    ├── analysis/             # Substitution matrices, key/mode analysis, optimisation
    ├── protein/              # Protein conservation/covariance/structure pipeline
    ├── pipeline/             # Analysis stages, named by analysis (identification,
    │                         #   mutability, substitution, position, covariance, …)
    └── viz/                  # Manuscript figure generation
scripts/                      # fetch_data.py, download_data.py, build_zenodo.sh
notebooks/                    # Tier 1/2/3 reproduction notebooks
Data/                         # Raw input corpora (fetched or symlinked; see Data section)
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
override with the `THESESSION_DATA` environment variable to use an out-of-tree data directory.
Run `python scripts/fetch_data.py` to populate `Data/` for the two fetchable sources.

A stable version of the data and code is archived for release. \[LINK\] The archive is split
into two tiers — see `README_DATA.md` for what each contains and how to fetch them
(`scripts/download_data.py`).
