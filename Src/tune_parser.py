import json
import re

import music21
import numpy as np

from global_variables import *
import pyabc


### Notes to self...
### pyabc comes up against some bugs, but it can read thesession data
### music21 needs to be fed in an exact format, which... (SOLVED)

### See setting 30643 - G# doesnt seem to be picked up by parser (SOLVED)

### Some errors come up.
### for setting 44997, there is a misplaced comma in the ABC string,
### after the B,2 in the 9th bar: "B,2,E EDE"
### for setting 38861, there is a random q
### 32371 - all sorts of weird stuff
### ^^^ IGNORED

### Grace notes (curly brackets, {}) & polyphonic (square brackets, [])
### (SOLVED)

### Polyphonic & multiple voices
### (SOLVED)

### music21 skips notes with pause markings, "Hd6" (I assume it's the H)
### e.g. https://thesession.org/tunes/17726#setting34482

### At this stage, there are only 305 tunes left where pyabc and music21
### don't agree on the length. I think I'll stop here for now...

### More issues!
### 33it [00:06,  4.74it/s]cannot expand Stream: badly formed repeats or repeat expressions
###     this is due to multiple voices; already caught these; in any case this will be ignored
### 37it [00:06,  5.66it/s]abcFormat: WARNING: Could not get pitch information from note:  q, assuming C
###     grace notes... already caught these
### 147it [00:32,  2.94it/s]abcFormat: WARNING: Could not get pitch information from note:  J, assuming C
###     glissando / slide; matching melody lengths between algorithms gets rid of this problem
### 161it [00:35,  5.09it/s]abcFormat: WARNING: Could not get pitch information from note:  X, assuming C
###     not sure what the X is supposed to be; matching melody lengths between algorithms gets rid of this problem
### 1166it [03:55,  5.24it/s]cannot expand Stream: badly formed repeats or repeat expressions
###     this one is missing a repeat; in any case, this is an error so it will be ignored
### 1220it [04:05,  1.82it/s]
###     this gets caught in an infinite loop since it is missing the start of the repeat


### ALSO, there are a bunch of tunes that don't have standard meter,
### but thesession doesn't allow non-standard meters, so they just put down
### whatever is closest. Need to check the actual meter in the abc header,
### and compare with the tune type



dict2header = {'area': 'A: ',
               'book': 'B: ',
               'composer': 'C: ',
               'discography': 'D: ',
               'file url': 'F: ',
               'group': 'T: ',
               'history': 'H: ',
               'instruction': 'I: ',
               'key': 'K: ',
               'mode': 'K: ',
               'unit note length': 'L: ',
               'meter': 'M: ',
               'rhythm': 'R: ',
               'type': 'R: ',
               'book': 'B: ',
               'transcription': 'Z: ',
               'username': 'Z: ',
               'tune_id': 'T: ',
               'setting_id': 'X: '
               }


###############################################################
### Code for ABC files and for pyabc Tune class


def parse_thesession_tune(data, alg='pyabc', trim_abc=True, expandRepeats=True):
    # Remove elements that are not used in analysis, but which
    # can cause problems for parsers
    if trim_abc:
        data['abc'] = remove_unnecessary_elements(data['abc'])

    # Parse header and body with pyabc
    if alg == 'pyabc':
        out = {}
        tune = pyabc.Tune(json=data)
        out['notestr'] = get_note_str(tune)
        out['abspitch'] = get_abspitch(tune)
        key, has_keys = get_key_idx_from_tune(tune)
        out['key'] = key
        out['has_keys'] = list(has_keys)
        out = account_for_ties(tune, out)
        return out

    # Parse body (tune) with music21
    elif alg == 'music21':
        abc_str = dict2abc(data)
        score = music21.converter.parseData(abc_str, format='abc')
        keys = ['bar_onset', 'onsets', 'dur', 'midi', 'rests', 'bars']
        vals = parse_score(score, expandRepeats=expandRepeats)
        out = {k:v for k, v in zip(keys, vals)}
    return out


### Converts the json data format to the typical ABC string
def dict2abc(json_data):
    # Add the setting ID first
    header = [f"X: {json_data['setting_id']}"]
    header += [f"{dict2header[k]} {v}" for k, v in json_data.items() if k in dict2header and k != 'setting_id']
    header.append("L: 1/8") # Should this be here? Are there tunes where this is not correct?
    header.append(json_data['abc'])
    return '\n'.join(header)


### Removes characters that correspond to elements that are unnecessary
### for analysis purposes (e.g. slurs (), pauses H), and which can actually
### lead to parsing problems
def remove_unnecessary_elements(abc):
    # Easy fix for slur-ending brackets and pauses
    for ele in ')H':
        abc = abc.replace(ele, '')

    # '(' is used for starting slurs, and also for duplets, triplets, etc.
    if '(' in abc:
        # If '(' is follwed by a number from 2-9, we need to ignore these
        if len(re.findall(r'\([2-9]', abc)) > 0:
            start = 0 # Start of the string
            startstop = [] # Container for start/stop points of string without '('
            for match in re.finditer(r'\(', abc):
                # Stopping point is given by the start of the '(' string
                stop = match.start()
                # Check if number follows '('
                if abc[stop + 1] in '23456789':
                    continue
                # At this point, we have identified a slur, and we append start/stop
                # indices to avoid the '('
                startstop.append((start, stop))
                # New starting point after the '(' 
                start = match.end()
            stop = len(abc)
            startstop.append((start, stop))
            abc = ''.join(abc[begin:end] for begin, end in startstop)
        # Otherwise use the cheap algorithm
        else:
            abc = abc.replace('(', '')
    return abc


### Looks for ties, and merges tied notes
### Currently doesn't add up durations; I use music21 for durations and onsets
def account_for_ties(tune, parsed_data):
    # First check for ties
    has_ties = sum([isinstance(token, pyabc.Tie) for token in tune.tokens])
    if not has_ties:
        return parsed_data

    idx2del = []    # container for indices to be deleted
    tie_active = False  # control switch for identifying notes to remove
    j = 0   # Counter for note indices
    for i, token in enumerate(tune.tokens):
        if isinstance(token, pyabc.Tie):
            tie_active = True
        elif isinstance(token, pyabc.Note):
            if tie_active:
                tie_active = False
                idx2del.append(j)
            j += 1

    # Remove tied notes
    parsed_data['notestr'] = ''.join([x for i, x in enumerate(parsed_data['notestr']) if i not in idx2del])
    parsed_data['abspitch'] = np.array([x for i, x in enumerate(parsed_data['abspitch']) if i not in idx2del])

    return parsed_data
                

### Check that repeat lines are opened and closed properly
def check_repeat_consistency(abc):
    repeat_starts = [m.start() for m in re.finditer(r'\|:', abc)]
    repeat_ends = [m.start() for m in re.finditer(r':\|', abc)]

    # Easiest check, there should be the same number of each
    if len(repeat_starts) != len(repeat_ends):
        return False

    # No repeats? No problem
    if len(repeat_starts) == 0:
        return True

    # Then check that the order is correct (no nested repeats)
    for i in range(len(repeat_starts)):
        if repeat_starts[i] > repeat_ends[i]:
            return False
        if (i + 1) != len(repeat_starts):
            if repeat_starts[i+1] < repeat_ends[i]:
                return False
    return True


def get_key_idx_from_tune(tune):
    key_by_note = [n.key.root.value for n in tune.notes]
    has_keys = set(key_by_note)
    return key_by_note[0], has_keys


def get_key_idx_from_mode(mode):
    key = {'G':7, 'D':2, 'A':9, 'B':11, 'C':0, 'F':5, 'E':4}
    return key.get(mode[0])


def tchroma_to_tnote(tchroma):
    if not len(tchroma):
        return ''
    key = np.array(list('ccddeeffgaabb'))
    return ''.join(key[tchroma])


def get_note_str(tune):
    return ''.join([n.note for n in tune.notes])


def get_pitch(tune):
    return np.array([n.pitch.value for n in tune.notes])


def get_abspitch(tune):
    return np.array([n.pitch.abs_value for n in tune.notes])


def transpose_pitch(pitch, key):
    return pitch - key


def convert_to_chroma(pitch):
    return pitch % 12


###############################################################
### music21 score parser


def parse_score(score, i=-1, expandRepeats=True):
    # music21 'Score' objects can contain different classes
    # as children. We are interested in parsing the 'Part' objects.
    # If no specific part index (i) is given, then find the first Part
    if i == -1:
        parts = [p for p in score.parts if isinstance(p, music21.stream.Part)]
        if len(parts) == 0:
            raise Exception("WARNING! No parts found in score!")
        part = parts[0]
    else:
        part = list(score.parts)[i]

    # music21 function can read repeat lines, and expand the score to reflect
    # the full melody as it should be performed
    if expandRepeats:
        part = part.expandRepeats()

    return parse_stream(part)


### Parses notes and rests
def parse_stream(stream):
    onset = 0
    bar_onset = []
    onsets = []
    dur = []
    midi = []
    rests = []
    bars = []
    prev_tie = ''
    tie = ''
    d0 = 0.
    for i, n in enumerate(stream.recurse()):
        if isinstance(n, music21.note.Note):
            # Ignore grace notes
            if n.quarterLength == 0:
                continue

            try:
                tie = n.tie.type
            except:
                tie = ''

            if tie == '':
                onsets.append(onset)
                dur.append(n.quarterLength)
                midi.append(n.pitch.midi)
                bar_onset.append(n._activeSiteStoredOffset)

            # If tie left bracket "[" is there add the duration of the note to d0
            # Only append onset once (if the tie was not open in the previous loop)
            elif tie == 'start':
                d0 += n.quarterLength
                if prev_tie != 'start':
                    onsets.append(onset)
                    bar_onset.append(n._activeSiteStoredOffset)

            # For ties, only add the pitch and duration at the end of the tie
            elif tie == 'stop':
                dur.append(d0 + n.quarterLength)
                midi.append(n.pitch.midi)

            # When tie gets closed, reset the duration to zero
            if tie != 'start':
                d0 = 0.

            # Save the value of tie
            prev_tie = tie

        if isinstance(n, music21.note.Rest):
            rests.append(len(onsets) - 1)

        if isinstance(n, music21.stream.Measure):
            # Annotate the bar using the index of the first note
            # in the bar
            bars.append(len(onsets))
                
        onset += n.quarterLength
    return bar_onset, onsets, dur, midi, rests, bars


###############################################################
### Parsing kern format


def parse_kern_music21(path):
    score = music21.converter.parse(path, format='humdrum')
    keys = ['bar_onset', 'onsets', 'dur', 'midi', 'rests', 'bars']
    dtype = [float, float, float, int, int, int]
    vals = parse_score(score)
    data = {k: np.array(v, d) for k, v, d in zip(keys, vals, dtype)}

    keysig, acc = get_keysig_kern(path)
    data['key'] = keysig
    data['accidentals'] = acc
    return data


def get_keysig_kern(path):
    try:
        text = ''.join(open(path, encoding='utf8', errors='ignore').readlines())
        keysig = extract_keysig_from_kern(text)
        acc = extract_accidentals_from_kern(text)
        return keysig, acc
    except Exception as e:
        print('KeySig', path, e)
        return ['', '']


def extract_accidentals_from_kern(text):
    for s in text.split('\n'):
        if not len(s):
            continue
        if s[:2] == '*k':
            return re.search(r"\[(.*?)\]", s).groups()[0]
    return ''


def extract_keysig_from_kern(text):
    possible_keys = [f"{a}{b}{c}" for a in "ABCDEFGabcdefg" for b in ['', 'b', '#', '-', '+'] for c in ['', 'm', '4', '5', '6']]
    for s in ''.join(text).split('\n'):
        if not len(s):
            continue
        if '*clef' in s:
            continue
        if s[0] == '*':
            for k in possible_keys:
                if s[1:len(k)+1] == k:
                    return k
    return ''



###############################################################
### generic functions for parsing tunes


### Converts a key from letter format into an integer,
### ranging from C = 0 to B = 11
def get_key_idx(key):
    return chromatic_map.get(key.upper(), np.nan)



