import torch
import os
from collections import defaultdict
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
import json
from collections import defaultdict
import pandas as pd
import math

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)

with open('data/didec/didec_image_specificity_bertjecls.json', 'r') as f:
    selfbertje = json.load(f)

with open('startingpoint_var.json', 'r') as f:
    spvar = json.load(f)

with open('gaze_variation_IOU.json', 'r') as f:
    gazeIOU = json.load(f)

with open('gaze_variation_SSD.json', 'r') as f:
    gazeSSD = json.load(f)

with open('onset_means.json', 'r') as f:
    onsets = json.load(f)

specs = []
spvars = []
gazevars = []
gazevars_SSD = []
means = []

for im in selfbleu:
    specs.append(selfbleu[im])
    spvars.append(spvar[im])
    gazevars.append(gazeIOU[im])
    gazevars_SSD.append(gazeSSD[im])
    means.append(onsets[im])

print(np.mean(spvars), np.min(spvars), np.max(spvars))

corr, pvalue = spearmanr(spvars, gazevars)
print('correlation sp var vs. gaze var IOU')
print(round(corr, 3), round(pvalue, 3))

corr, pvalue = spearmanr(gazevars, specs)
print('correlation gaze var IOU vs. img spec')
print(round(corr, 3), round(pvalue, 3))

corr, pvalue = spearmanr(gazevars, means)
print('correlation gaze var IOU vs. mean onset')
print(round(corr, 3), round(pvalue, 3))

corr, pvalue = spearmanr(gazevars, gazevars_SSD)
print('correlation gaze var IOU vs. SSD')
print(round(corr, 3), round(pvalue, 3))

min_calc_im = ''
max_calc_im = ''
min_calc_val = math.inf
max_calc_val = -math.inf

for img in gazeIOU:
    cur_im_score = gazeIOU[img]
    if cur_im_score < min_calc_val:
        min_calc_val = cur_im_score
        min_calc_im = img

    if cur_im_score > max_calc_val:
        max_calc_val = cur_im_score
        max_calc_im = img

print('min calc gaze var:', min_calc_im, str(min_calc_val))
print('max calc gaze var:', max_calc_im, str(max_calc_val))
