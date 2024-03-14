import torch
import os
from collections import defaultdict
from scipy.spatial import distance
import numpy as np
import json

image_sentence_reps = defaultdict(list)

for root, dirs, files in os.walk('data/didec/bertdutch_sentencereps_pooler/'):

    for f in files:
        reps_file = os.path.join(root, f)
        _, _, ppn, im = f.split('.')[0].split('_')

        sentence_reps = torch.load(reps_file)
        image_sentence_reps[im].append(sentence_reps)

img_specificity = defaultdict()

for img in image_sentence_reps:
    reps = image_sentence_reps[img]
    sims = []

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            # print(i, j)
            sim = 1 - distance.cosine(reps[i].detach(), reps[j].detach())
            sims.append(sim)

    img_spec = np.mean(sims)
    print(img, round(img_spec, 3))

    img_specificity[img] = img_spec

with open('data/didec/didec_image_specificity_bertjepooler.json', 'w') as f:
    json.dump(img_specificity, f)
