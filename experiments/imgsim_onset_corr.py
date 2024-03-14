import json
import math
import pickle
from collections import defaultdict
import numpy as np
from torch import nn
import torch
from scipy.stats import spearmanr

with open('clip_im_sims.pickle', 'rb') as f:
    sims = pickle.load(f)

with open('data/didec/clip_preprocessed_didec_images_cpu.pickle', 'rb') as f:
    clip_preprocessed_images = pickle.load(f)

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)

datapaths = {'train': 'data/didec/split_train.json',
             'val': 'data/didec/split_val.json',
             'test': 'data/didec/split_test.json'}

dataset_param = defaultdict(dict)

for split in datapaths:

    print(split)
    path = datapaths[split]

    with open(path, 'r') as f:
        datasplit = json.load(f)

    count_aligned = 0

    for im in datasplit:

        dataset_param[im] = {'imgspec': selfbleu[im]}

        onset_list = []

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
            onset_list.append(delay)

        dataset_param[im].update({'mean_ons': np.mean(onset_list)})

    print(split, count_aligned)

print()

corr_sims = []
corr_loss = []
corr_specs = []

for im in dataset_param:
    others = set(dataset_param.keys()) - {im}
    for im2 in others:
        pair_sim = sims[im][im2]
        pair_onsetloss = abs(dataset_param[im]['mean_ons'] - dataset_param[im2]['mean_ons'])
        pair_specloss = abs(dataset_param[im]['imgspec'] - dataset_param[im2]['imgspec'])

        corr_sims.append(pair_sim)
        corr_loss.append(pair_onsetloss)
        corr_specs.append(pair_specloss)

print(len(corr_sims), len(corr_loss))

corr, pvalue = spearmanr(corr_sims, corr_loss)
print('correlation image sims vs. abs onset loss')
print(round(corr, 4), round(pvalue, 4))

corr, pvalue = spearmanr(corr_sims, corr_specs)
print('correlation image sims vs. abs img spec loss')
print(round(corr, 4), round(pvalue, 4))


corr, pvalue = spearmanr(corr_loss, corr_specs)
print('correlation abs onset loss vs. abs img spec loss')
print(round(corr, 4), round(pvalue, 4))



