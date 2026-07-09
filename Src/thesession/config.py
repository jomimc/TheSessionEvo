import os
from pathlib import Path

from Bio.Data.IUPACData import protein_letters
import numpy as np

__all__ = [
    "PATH_BASE", "PATH_DATA", "PATH_FIG", "PATH_FIG_DATA", "PATH_CACHE",
    "PATH_MMSEQS", "PATH_PROTEIN",
    "N_PROC",
    "letters", "chromatic_notes", "chromatic_map", "position_key", "letter_key",
    "note_list", "note_map",
    "MMSEQS_BIN",
    "MODES", "MODE_DIFF", "MODE_COMPAT",
    "METER_LIST", "SUBDIV_METER", "HIERARCHY", "END_POS",
    "DANCE_LIST", "SUBDIV_DANCE", "HIERARCHY_DANCE",
]


### Path to directory
# Repo root = two levels up from this file (Src/thesession/config.py -> repo root)
PATH_BASE = Path(__file__).resolve().parents[2]

# Raw input corpora. Defaults to <repo>/Data; override with THESESSION_DATA
# for an out-of-tree data directory (e.g. the Zenodo download location).
PATH_DATA = Path(os.environ.get("THESESSION_DATA", PATH_BASE / "Data"))

PATH_FIG = PATH_BASE.joinpath("Figures")
PATH_FIG_DATA = PATH_BASE.joinpath("FigureData")
PATH_CACHE = PATH_BASE.joinpath("Cache")
PATH_MMSEQS = PATH_BASE.joinpath("MMseqs")
PATH_PROTEIN = PATH_BASE.joinpath("ProteinData")

### Number of processors used in multiprocessing
N_PROC = 8

### Set of letters to be used with alignment algorithms
letters = np.array(list(protein_letters + 'X'))

### Basic solfege letter notation
chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chromatic_map = {n:i for i, n in enumerate(chromatic_notes)}
position_key = {a:i for i, a in enumerate(chromatic_notes)}
position_key['-'] = 100

### Mapping between solfege and protein letters
letter_key = {b:a for a, b in zip(chromatic_notes, letters)}
letter_key['-'] = '-'

### Mapping between letter and chroma in Savage et al's Bronson data
note_list = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G','a',  'A', 'b', 'B']
note_map = {n:i for i, n in enumerate(note_list)}
note_map['f'] = 5
note_map['c'] = 0


### Names (or paths to binaries) of software to run
MMSEQS_BIN = 'mmseqs'


### Modes
MODES = {'major': np.array([0, 2, 4, 5, 7, 9, 11]),
         'mixolydian': np.array([0, 2, 4, 5, 7, 9, 10]),
         'minor': np.array([0, 2, 3, 5, 7, 8, 10]),
         'dorian': np.array([0, 2, 3, 5, 7, 9, 10])}

MODE_DIFF = {(a, b): {MODES[a][i]: MODES[b][i] for i in np.where(MODES[a] != MODES[b])[0]}
             for a in MODES for b in MODES if a != b}

#MODE_MAP_FN = {(a,b): lambda x: x}

MODE_COMPAT = {'major': ['major', 'major pentatonic'],
               'mixolydian': ['mixolydian', 'major pentatonic'],
               'minor': ['minor', 'minor pentatonic'],
               'dorian': ['dorian', 'minor pentatonic']}


### Meters
METER_LIST = ['4/4', '6/8', '2/4', '9/8', '12/8']
SUBDIV_METER = {m:d for d, m in zip([8, 6, 4, 9, 12], METER_LIST)}

HIERARCHY = {'4/4':[0,2,1,2,0,2,1,2],
             '2/4':[0,1,0,1],
             '6/8':[0,2,1,0,2,1],
             '9/8':[0,2,1,0,1,2,0,1,2],
             '12/8':[0,2,1,0,1,2,0,1,2,0,1,2],
             '3/4':[0,2,1,2,1,2]}
END_POS = {'4/4':[0,0,0,0,0,0,0,1],
           '2/4':[0,0,0,1],
           '6/8':[0,0,0,0,0,1],
           '9/8':[0,0,0,0,0,0,0,0,1],
           '12/8':[0,0,0,0,0,0,0,0,0,0,0,1],
           '3/4':[0,0,0,0,0,1]}

### Dance types
DANCE_LIST = ['reel', 'jig', 'polka', 'hornpipe']
SUBDIV_DANCE = {m:d for d, m in zip([8, 6, 4, 8], DANCE_LIST)}

HIERARCHY_DANCE = {'reel':[0,2,1,2,0,2,1,2],
                   'hornpipe':[0,2,1,2,0,2,1,2],
                   'polka':[0,1,0,1],
                   'jig':[0,2,1,0,2,1],
                   'slip jig':[0,2,1,0,1,2,0,1,2],
                   'slide':[0,2,1,0,1,2,0,1,2,0,1,2],
                   'waltz':[0,2,1,2,1,2]}



