import os

import numpy as np
from scipy.io import wavfile


def load_dataset(path_to_data, dict_labels=None):
    path_listdir = os.listdir(path_to_data)
    raw_data = []
    path_listdir = path_listdir[:int(len(path_listdir) / 1)]

    for sound in path_listdir:
        ret, sound_data = wavfile.read(path_to_data + sound)
        if dict_labels is None:
            raw_data.append(np.array(sound_data))

        else:
            raw_data.append([np.array(sound_data), dict_labels[sound]])
    # DEBUG
    # print(len(raw_data))
    np.savetxt(path_to_data + 'filenames.txt',path_listdir, fmt = '%s')
    return raw_data


def find_longest(data):
    max_length = 0
    for sound in data:
        if sound.shape[0] > max_length:
            max_length = sound.shape[0]
    print(max_length)
    return max_length


def padding_data(data, length):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    pad_sound = []
    for sound in data:
        if sound.shape[0] < length:
            if sound.shape[0] % 2 == 0:
                pad_sound.append(np.pad(sound, int((length - sound.shape[0]) / 2), 'reflect'))
            else:
                incorrect_sound = np.pad(sound, int((length - sound.shape[0]) / 2), 'reflect')
                correct_sound = np.append(incorrect_sound, 0)
                pad_sound.append(correct_sound)
    return pad_sound


def get_random_segment(sound):
    length_segment = 48000
    if len(sound) <= length_segment:
        pad_number = int(((length_segment - len(sound)) / 2 +1))
        # DEBUG
        # print(len(sound) + pad_number*2)
        sound = np.pad(sound, pad_number, 'reflect')
    random_number = np.random.randint(0, abs(len(sound) - length_segment))
    return sound[random_number:random_number+ length_segment]

