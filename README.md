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
- Note mutability correlates with tonal hierarchy (cognitive constraints on transmission)
- Substitution rates mirror interval distributions (motor constraints on melodic motion)
- Evolutionary stability tracks metrical hierarchy (skeletal melody encodes tune identity)
- Repetition structure is conserved across variants (signature of cultural selection)

## Requirements

- Python 3.12+
- [MMseqs2](https://github.com/soedinglab/MMseqs2) (must be on PATH as `mmseqs`)
- Python packages: `music21`, `biopython`, `pandas`, `numpy`, `scipy`, `scikit-learn`,
  `statsmodels`, `matplotlib`, `seaborn`, `networkx`

## Usage

**Interactive analysis:**
```bash
cd Src/
python start_ipython.py
```
This pre-loads all modules and common data objects for interactive exploration.

**Full pipeline:**
```bash
cd Src/
python run_pipeline.py
```
The pipeline runs end-to-end: parsing ABC notation → extracting tune parts → running
MMseqs2 all-vs-all alignment → statistical analysis → figure generation.

Results are cached in `Cache/`; set `redo=True` in individual steps to recompute.
Generated figures are saved to `Figures/`.

## Project Structure

```
Src/
├── run_pipeline.py           # Main analysis orchestrator
├── start_ipython.py          # Interactive session setup
└── thesession/
    ├── config.py             # Global constants, paths, musical mode definitions
    ├── utils.py              # Sequence conversion, statistics utilities
    ├── io/                   # Data loading: ABC notation, FASTA, DataFrames
    ├── alignment/            # Pairwise and rhythm-aware part alignment
    ├── structure/            # Tune part separation (A/B sections)
    ├── analysis/             # Substitution matrices, key/mode analysis, optimisation
    └── viz/                  # Manuscript figure generation
Cache/                        # Cached intermediate data (pkl, npy)
FigureData/                   # Numerical data backing each figure
Figures/                      # Generated PDF/PNG figures
MMseqs/                       # MMseqs2 FASTA inputs and result tables
```

## Data

Input data are drawn from three sources:

- **TheSession** — Irish/Celtic tunes in ABC notation. \[https://github.com/adactio/TheSession-data \]
- **Meertens Tune Collections** — Dutch folk music collection. \[https://www.liederenbank.nl/mtc \]
- **Savage et al.** — Manually-aligned melody pairs from the Bronson ballad collection. \[https://github.com/pesavage/melodic-evolution/tree/master/data \]

Raw data paths are configured in `Src/thesession/config.py`.

A stable version of the data and code is archived on Zenodo. \[LINK\]

