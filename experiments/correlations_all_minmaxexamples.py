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

with open('startingpoint_var.json', 'r') as f:
    spvar = json.load(f)

with open('gaze_variation_IOU.json', 'r') as f:
    gazeIOU = json.load(f)

with open('onset_means.json', 'r') as f:
    onsets = json.load(f)

metrics = {'img_spec': selfbleu, 'SPvar': spvar, 'GazeVar': gazeIOU, 'Onset': onsets}

for mt in metrics:

    metric = metrics[mt]

    min_actual_im = ''
    max_actual_im = ''
    min_actual_val = math.inf
    max_actual_val = -math.inf

    for img in metric:
        img_score = metric[img]
        if img_score < min_actual_val:
            min_actual_val = img_score
            min_actual_im = img

        if img_score > max_actual_val:
            max_actual_val = img_score
            max_actual_im = img

    print(mt)
    print('min actual:', min_actual_im, str(round(min_actual_val, 3)))
    print('max actual:', max_actual_im, str(round(max_actual_val, 3)))
    print()
