import pickle
import numpy as np
import torch
import json
from scipy.spatial import distance
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from collections import defaultdict

from Metric_IOU_Gaze import Metric

mm = Metric()

with open('data/didec/bbox_preprocessed_didec_fx2SAMcropped_B32_cpu.pickle', 'rb') as f:
    bbox_preprocessed_fxs = pickle.load(f)

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)

sequences = defaultdict(list)
count_tp = 0

for tp in bbox_preprocessed_fxs:

    count_tp += 1
    if count_tp % 100 == 0:
        print(count_tp)

    ppn, im = tp
    bboxes = bbox_preprocessed_fxs[tp]

    sequences[im].append(bboxes)

all_img_IOUs = []
imgs_specs = []
all_img_IOUs_dict = defaultdict(float)

count = 0

for img in sequences:

    count += 1
    print('count img', count)

    img_IOUs = []
    count_seqs = len(sequences[img])

    for c in range(count_seqs):
        if c == count_seqs - 1:
            print(c)
        set_others = set(range(count_seqs)) - {c}
        seqs_others = [sequences[img][j] for j in set_others]

        img_ppn_IOUs = []

        for so in seqs_others:
            pair_IOU = mm.new_metric(sequences[img][c], so)

            img_ppn_IOUs.append(pair_IOU)

        single_ppn_meanIOU = np.mean(img_ppn_IOUs)
        img_IOUs.append(single_ppn_meanIOU)

    img_IOU = np.mean(img_IOUs)

    all_img_IOUs.append(img_IOU)
    all_img_IOUs_dict[img] = img_IOU
    imgs_specs.append(selfbleu[img])

corr, pvalue = spearmanr(all_img_IOUs, imgs_specs)
print('MEAN full set correlation img avg fx sequence IOU vs. ling variation bleu-2')
print(round(corr, 4), round(pvalue, 4))

with open('gaze_variation_IOU.json', 'w') as f:
    json.dump(all_img_IOUs_dict, f)
