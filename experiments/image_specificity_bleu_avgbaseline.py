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

with open(datapaths['train'], 'r') as f:
    datasplit = json.load(f)

specs = []

for im in datasplit:
    specs.append(selfbleu[im])

spec_train_mean = np.mean(specs)
print(round(spec_train_mean, 4))

for split in ['val', 'test']:

    count = 0
    split_losses = []

    print(split)
    path = datapaths[split]

    dataset_param = defaultdict(dict)

    with open(path, 'r') as f:
        datasplit = json.load(f)

    for im in datasplit:
        target = selfbleu[im]
        loss = abs(target - spec_train_mean)
        split_losses.append(loss)

    print(split, round(np.mean(split_losses), 4))



