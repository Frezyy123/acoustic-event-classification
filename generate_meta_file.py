import os
import re

import numpy as np

path_to_files = '/home/alexandr/data/test/'
text_labels = {'background': '0', 'bags': '1', 'door': '2', 'keyboard': '3', 'knocking': '4', 'ring': '5', 'speech': '6',
               'tool': '7', 'unknown': '8'}

test_sound = os.listdir(path_to_files)

meta_information = []
unknown_meta_information = []
for sound in test_sound:

    splitted_name = re.split('_\d*', sound)
    if splitted_name[0] != 'unknown':

        meta_information.append([sound, text_labels[splitted_name[0]]])
    else:
        unknown_meta_information.append([sound, text_labels[splitted_name[0]]])



print(meta_information)

np.savetxt('meta-test.txt',meta_information,fmt='%s')
np.savetxt('unknown-meta.txt',unknown_meta_information,fmt='%s')