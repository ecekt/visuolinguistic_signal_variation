from scipy.stats import normaltest
import json
import numpy as np

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)

datapaths = {'train': 'data/didec/split_train.json',
             'val': 'data/didec/split_val.json',
             'test': 'data/didec/split_test.json'}

full_imgspec = []

for split in datapaths:

    print(split)
    path = datapaths[split]

    split_imgspec = []

    with open(path, 'r') as f:
        datasplit = json.load(f)

    for im in datasplit:
        split_imgspec.append(selfbleu[im])
        full_imgspec.append(selfbleu[im])

    k2, p = normaltest(split_imgspec)
    print(len(split_imgspec), split, 'normality', k2, p)

k2, p = normaltest(full_imgspec)
print(len(full_imgspec), 'full normality', k2, p)
