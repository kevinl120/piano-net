import glob
import numpy as np
import os

from keras.utils import to_categorical


def pitch_to_num(note):
    """Takes a note of the form {PITCH}{ACCIDENTAL(s)}{OCTAVE} and turns it 
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


def from_fingering_file(start_file=0):
    files = glob.glob('/kaggle/input/pianofingerings/FingeringFiles/*-1_fingering.txt')
    files.sort()

    x_train = []
    y_train = []

    for path in files[start_file:]:
        with open(path) as f:
            all_lines = [line for line in f.readlines() if line[0] != '/']
            x_file = []
            y_file = []
            for line in all_lines:
                l = line.split('\t')
                x_file.append([float(l[1]), float(l[2]), pitch_to_num(l[3])])

                fingering = [int(x) for x in l[7].split('_')][0]
                fingering = fingering + (5 if fingering < 0 else 4)
                y_file.append(fingering)
            for i in range(len(y_file) - 30):
                x_train.append(np.array(x_file[i:i+30]))
                y_train.append(to_categorical(y_file[i+15], num_classes=10))
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train