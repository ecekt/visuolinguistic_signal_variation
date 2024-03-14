import json
import math
import pickle
from collections import defaultdict
import numpy as np
from torch import nn
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import spacy
from spacy.lang.nl.examples import sentences

nlp = spacy.load("nl_core_news_lg")

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

onset_list = []
noun_count_list = []

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
                text = []

                if len(lines) == 0:
                    print('empty', f)
                else:
                    count_aligned += 1
                    print(count_aligned)

                    for line in lines:
                        if count_line == 0:
                            delay = line['start']

                        count_line += 1

                        text.append(line['text'])

                    text_str = ' '.join(text)
                    print(text_str)

                    doc = nlp(text_str)

                    noun_count = len(lines)  # word count actually

                    # for token in doc:
                        # if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                        #     if token.text.lower() not in ['aantal', 'uh', 'paar'] and 'foto' not in token.text.lower():
                        #         noun_count += 1

            print('speech onset', delay, '\n')
            onset_list.append(delay)
            noun_count_list.append(noun_count)

corr, pvalue = spearmanr(onset_list, noun_count_list)
print(split, 'correlation target onset vs. noun count')
print(round(corr, 4), round(pvalue, 4))


# full set correlation target onset vs. word count
# -0.1014 0.0

## full set correlation target onset vs. noun count
# -0.0274 0.0632

# train correlation target onset vs. noun count
# -0.041 0.0129
#
# val correlation target onset vs. noun count
# -0.0052 0.9129
#
# test correlation target onset vs. noun count
# 0.0583 0.2172

