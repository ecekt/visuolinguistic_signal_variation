import torch
import os
from collections import defaultdict
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
import json

with open('data/didec/didec_image_specificity_bertjecls.json', 'r') as f:
    bertje_cls = json.load(f)

min_spec = float('inf')
max_spec = -float('inf')
min_im = ''
max_im = ''

for im in bertje_cls:
    spec = bertje_cls[im]
    if spec < min_spec:
        min_spec = spec
        min_im = im
    if spec > max_spec:
        max_spec = spec
        max_im = im

print('max', max_im, max_spec)
print('min', min_im, min_spec)

datapaths = {'train': 'data/didec/split_train.json',
                 'val': 'data/didec/split_val.json',
                 'test': 'data/didec/split_test.json'}

print('max sentences', max_im)

for split in datapaths:

    print(split)
    path = datapaths[split]

    with open(path, 'r') as f:
        datasplit = json.load(f)

    if max_im in datasplit:
        for ppn in datasplit[max_im]:

            alignment_file = 'data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + max_im + '.json'

            delay = 0.0
            count_line = 0

            with open(alignment_file, 'r') as j:
                lines = json.load(j)

                text = []

                if len(lines) == 0:
                    print('empty', f)
                else:
                    for line in lines:
                        if count_line == 0:
                            delay = line['start']

                        count_line += 1
                        text.append(line['text'])

            print(round(delay, 3), ' '.join(text))
        print()

print('min sentences', min_im)

for split in datapaths:

    print(split)
    path = datapaths[split]

    with open(path, 'r') as f:
        datasplit = json.load(f)

    if min_im in datasplit:
        for ppn in datasplit[min_im]:

            alignment_file = 'data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + min_im + '.json'

            delay = 0.0
            count_line = 0

            with open(alignment_file, 'r') as j:
                lines = json.load(j)

                text = []

                if len(lines) == 0:
                    print('empty', f)
                else:
                    for line in lines:
                        if count_line == 0:
                            delay = line['start']

                        count_line += 1
                        text.append(line['text'])

            print(round(delay, 3), ' '.join(text))
        print()
