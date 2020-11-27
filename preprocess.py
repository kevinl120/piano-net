import glob
import numpy as np
import os
import pandas as pd
import tqdm

from keras.utils import to_categorical


def pitch_to_num(note):
    """Takes a spelled pitch of the form {PITCH}{ACCIDENTAL(s)}{OCTAVE} and turns it 
    into the corresponding key number on the piano. E.g. A0=1, C4=40, F##4=47
    """
    pitch_map = {'A': 13, 'B': 15, 'C': 4, 'D': 6, 'E': 8, 'F': 9, 'G': 11}
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


def get_finger(f, two_hand=False):
    f = str(f)
    finger = int([x for x in f.split('_')][0])
    if two_hand:
        finger = finger + (5 if finger < 0 else 4)
    else:
        finger = abs(finger) - 1
    return finger


def note_is_black(n):
    return n%12==2 or n%12==4 or n%12==7 or n%12==9 or n%12==11


def from_fingering_file(context=5, two_hand=False):
    files = glob.glob('data/FingeringFiles/*_fingering.txt')
    files.sort()

    x_train = []
    y_train = []
    
    col_names = ['noteID', 'onsetTime', 'offsetTime', 'pitch', 'onsetVelocity', 'offsetVelocity', 'channel', 'finger']
    
#     for path in files[:1]:
    for path in tqdm.tqdm(files[:]):
        df = pd.read_csv(path, sep='\t', header=0, names=col_names)
        df['pitch'] = df['pitch'].apply(pitch_to_num)
        df['finger'] = df['finger'].apply(lambda x: get_finger(x, two_hand=two_hand))
        rh_df = df[df['channel'] == 0]
        
        padding = pd.DataFrame([[0]*len(col_names) for i in range(context)], columns=col_names)
        df = pd.concat([padding, rh_df, padding])
        df.reset_index(drop=True, inplace=True)
        x_file = df[['onsetTime', 'offsetTime', 'pitch']]
        y_file = df[['finger']]
        
        for i in range(len(df) - (2 * context)):
            x_data = x_file.iloc[i:i+(2*context)+1].to_numpy()
            y_data = to_categorical(y_file.iloc[i+context].to_numpy()[0], num_classes=(10 if two_hand else 5))
            
            mask = ((x_data[:, 2 ] == 0) * 88)
            
            octave = 0
            while octave < 8:
                x_data_transformed = x_data.copy()
                x_data_transformed[:, 2] = x_data[:, 2] - ((np.min(x_data[:, 2] + mask) // 12) * 12)
                x_data_transformed[:, 2] = x_data_transformed[:, 2] + octave * 12
                x_data_transformed[:, 2] = x_data_transformed[:, 2] * (x_data_transformed[:, 1] != 0)
                octave += 1
                if np.max(x_data_transformed) > 88:
                    break
            
                x_train.append(x_data_transformed)
                y_train.append(y_data)
        
    return np.array(x_train), np.array(y_train)