import os

import numpy as np
from scipy.io import wavfile


def load_dataset(path_to_data):
    path_listdir = os.listdir(path_to_data)

    raw_data = []
    path_listdir = path_listdir[:int(len(path_listdir) / 10)]

    for sound in path_listdir:
        ret, sound_data = wavfile.read(path_to_data + sound)
        raw_data.append(np.array(sound_data))

    print(len(raw_data))
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
    length_segment = 16000
    if len(sound) < length_segment:
        print(len(sound))

        sound = np.pad(sound, int((length_segment - len(sound) / 2)), 'reflect')

    random_number = np.random.randint(0, len(sound) - length_segment)
    return sound[random_number:length_segment + random_number]
