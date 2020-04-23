import glob
import keras
import numpy as np
import os

import pdb


def pitch_to_num(note):
    """Takes a note of the form [PITCH][ACCIDENTAL(s)][OCTAVE] and turns it 
    into the corresponding key number on the piano. E.g. A1=1, C4=40, F##4=47
    """
    pitch_map = {'A': 1, 'B': 3, 'C': 4, 'D': 6, 'E': 8, 'F': 9, 'G': 11}
    res = pitch_map[note[0]]
    if note[1] == '#':
        res += 1
        if note[2] == '#':
            res += 1
    if note[1] == 'b':
        res -= 1
        if note[2] == 'b':
            res -= 1
    res += 12 * (ord(note[-1]) - ord('1'))
    return res


def from_fingering_file():
    files = glob.glob('data/FingeringFiles/*-1_fingering.txt')
    files.sort()

    x_train = []
    y_train = []

    for path in files:
        # skip over 005-1 since that has a triple finger switch
        if path == 'data/FingeringFiles/005-1_fingering.txt':
            path = 'data/FingeringFiles/005-2_fingering.txt'

        with open(path) as f:
            x_file = []
            y_file = []
            for line in f.readlines():
                if line[0] == '/':
                    continue
                l = line.split('\t')
                x_file.append([float(l[1]), float(l[2]), pitch_to_num(l[3])])

                fingering = [int(x) for x in l[7].split('_')]
                if len(fingering) == 1:
                    fingering.append(fingering[0])
                fingering = map(lambda x: x + (5 if x < 0 else 4), fingering)
                y_file.append(list(fingering))
            y_file = keras.utils.to_categorical(y_file)

            x_train.append(x_file)
            y_train.append(y_file)
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    pdb.set_trace()
            

from_fingering_file()