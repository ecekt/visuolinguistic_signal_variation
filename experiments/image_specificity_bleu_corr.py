import torch
import os
from collections import defaultdict
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
import json
from collections import defaultdict

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)

datapaths = {'train': 'data/didec/split_train.json',
                 'val': 'data/didec/split_val.json',
                 'test': 'data/didec/split_test.json'}

onsets_full = defaultdict(list)

for split in datapaths:

    print(split)
    path = datapaths[split]

    dataset_param = defaultdict(dict)

    with open(path, 'r') as f:
        datasplit = json.load(f)

    count_aligned = 0

    for im in datasplit:
        for ppn in datasplit[im]:

            alignment_file = 'data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + im + '.json'

            with open(alignment_file, 'r') as j:
                lines = json.load(j)

                delay = 0
                count_line = 0

                if len(lines) == 0:
                    print('empty', f)
                else:
                    count_aligned += 1
                    print(count_aligned)

                    for line in lines:
                        if count_line == 0:
                            delay = line['start']

                        count_line += 1

            print('speech onset', delay, '\n')

            onsets_full[im].append(delay)

onsets_avg = []
onsets_std = []
specs = []

onsets_mean_dict = dict()

for im in onsets_full:
    onsets_avg.append(np.mean(onsets_full[im]))
    onsets_mean_dict[im] = np.mean(onsets_full[im])
    onsets_std.append(np.std(onsets_full[im]))
    specs.append(selfbleu[im])
    print(im, np.mean(onsets_full[im]), np.std(onsets_full[im]), selfbleu[im])

print(len(onsets_avg))

corr, pvalue = spearmanr(onsets_avg, specs)
print('correlation avg onset vs. img spec')
print(round(corr, 3), round(pvalue, 3))

corr, pvalue = spearmanr(onsets_std, specs)
print('correlation std onset vs. img spec')
print(round(corr, 3), round(pvalue, 3))

corr, pvalue = spearmanr(onsets_avg, onsets_std)
print('correlation avg onset vs. std onset')
print(round(corr, 3), round(pvalue, 3))

with open('onset_means.json', 'w') as f:
    json.dump(onsets_mean_dict, f)
