import os
from math import sqrt


FREQ = 16000
WINDOW_MS = 0.02
path_to_file = '/home/alexandr/data/audio/'
path_listdir = os.listdir(path_to_file)
window_length = int(WINDOW_MS * FREQ)
threshold = 200

def calculate_RMS(frame):
    summ = 0
    for amp in frame:
        summ += amp ** 2
    RMS = sqrt(summ / len(frame))
    return RMS


def detect_event(data):
    frames = []
    threshold_detector = 2000
    overall_RMS = 0
    for i in range(int(len(data) / window_length)):
        frame = data[i * window_length:window_length * (1 + i)]
        overall_RMS += calculate_RMS(frame)
        frames.append(data[i * window_length:window_length * (1 + i)])
    if overall_RMS > threshold_detector:
        isEvent = True
    else:
        isEvent = False

    return overall_RMS, isEvent
