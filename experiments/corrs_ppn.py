import json
import pickle
from collections import defaultdict
import numpy as np
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

original_data = dict()

for split in datapaths:

    print(split)
    path = datapaths[split]

    dataset_param = defaultdict(dict)

    with open(path, 'r') as f:
        datasplit = json.load(f)

    count_aligned = 0

    for im in datasplit:

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
            dataset_param[ppn][im] = delay

    print(split, count_aligned)
    original_data[split] = dataset_param

print('data lengths', len(original_data['train']), len(original_data['val']), len(original_data['test']), '\n')

splits = ['train', 'val', 'test']

img_specs = defaultdict(list)
ppn_onsets = defaultdict(list)

for spl in splits:

    print(split)
    ppn_count = 0

    for ppn in original_data[spl]:

        ppn_count += 1

        for im in original_data[spl][ppn]:
            img_specs[ppn].append(selfbleu[im])
            ppn_onsets[ppn].append(original_data[spl][ppn][im])

count = 0
count_sig = 0
count_neg = 0

for ppn in ppn_onsets:

    count += 1

    corr, pvalue = spearmanr(img_specs[ppn], ppn_onsets[ppn])
    print(ppn, 'correlation img spec vs. target ppn onset')
    print(round(corr, 4), round(pvalue, 4))

    if pvalue < 0.05:
        print('sigcorr')
        count_sig += 1

    if corr < 0:
        count_neg += 1

    print()

print('all', count, 'sig', count_sig, 'neg', count_neg)
