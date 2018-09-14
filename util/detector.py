import os
from math import sqrt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense, Activation, LSTM, Dropout, Conv2D, Flatten, np
from speechpy.feature import lmfe

FREQ = 16000
WINDOW_MS = 0.02
path_to_file = './audio/'
path_listdir = os.listdir(path_to_file)
window_length = int(WINDOW_MS * FREQ)
threshold = 200

def calculate_RMS(frame):
    summ = 0
    for amp in frame:
        summ += amp ** 2
    RMS = sqrt(summ / len(frame))
    return RMS

class Detector:

    def __init__(self, model = 2):
        self.model = Sequential()
        if model == 2:
            self.model.add(LSTM(256, input_shape=(298,40), return_sequences=True))

            self.model.add(LSTM(128,  return_sequences=False))

            self.model.add(Dense(2))
            self.model.add(Activation('sigmoid'))
            self.model.load_weights('./models/detector_model_2')
        else:
            self.model.add(LSTM(256, input_shape=(298, 40), return_sequences=False))

            self.model.add(Dense(2))
            self.model.add(Activation('sigmoid'))
            self.model.load_weights('./models/detector_model_2')

    def detect_event(self, data):


        feature = lmfe(data.astype(float), 16000)
        isEvent = np.argmax(self.model.predict(feature.reshape(-1, 298, 40)),axis = 1) == 1


        return  isEvent
