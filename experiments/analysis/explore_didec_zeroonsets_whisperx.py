import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math

datapaths = {'train': '../data/didec/split_train.json',
             'val': '../data/didec/split_val.json',
             'test': '../data/didec/split_test.json'}

count_zerostarters = 0

for split in datapaths:

    print(split)
    path = datapaths[split]

    with open(path, 'r') as f:
        datasplit = json.load(f)

    count_aligned = 0
    speech_onsets = []

    for im in datasplit:

        image_onsets = []

        for ppn in datasplit[im]:

            alignment_file = '../data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + im + '.json'

            with open(alignment_file, 'r') as j:
                lines = json.load(j)

                onset = 0

                if len(lines) == 0:
                    print('empty', f)
                else:
                    count_aligned += 1
                    print(count_aligned)

                    for line in lines:

                        onset = line['start']
                        break

            print('speech onset', onset, '\n')

            if float(onset) == 0:
                print('zerostarter', im, ppn)
                count_zerostarters += 1

print(count_zerostarters)
