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
            all_lines = [line for line in f.readlines() if line[0] != '/']
            x_file = []
            y_file = []
            for line in all_lines:
                l = line.split('\t')
                x_file.append([float(l[1]), float(l[2]), pitch_to_num(l[3])])

                fingering = [int(x) for x in l[7].split('_')]
                if len(fingering) == 1:
                    fingering.append(fingering[0])
                fingering = map(lambda x: x + (5 if x < 0 else 4), fingering)
                y_file.append(list(fingering))
            for i in range(len(y_file)//10 - 2):
                x_train.append(np.array(x_file[i*10:i*10+30]))
                y_train.append(keras.utils.to_categorical(y_file[i*10+10:i*10+20], num_classes=10))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (-1, 200))

    return x_train, y_train

# from_fingering_file()