import json
import math
import pickle
from collections import defaultdict
import numpy as np
from torch import nn
import torch
from scipy.stats import spearmanr

with open('clip_im_sims_NONE.pickle', 'rb') as f:
    sims = pickle.load(f)

with open('data/didec/clip_NONE_preprocessed_didec_images_cpu.pickle', 'rb') as f:
    clip_preprocessed_images = pickle.load(f)

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)


def retrieve_closest(sample_im, training_ims, k_count):

    sims_all = sims[sample_im]
    sorted_sims = sorted(sims_all.items(), key=lambda x: x[1], reverse=True)

    top_k = []
    count = 0

    for tp in sorted_sims:
        if tp[0] in training_ims:
            top_k.append(tp)
            count += 1
        if count == k_count:
            break

    return top_k


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

        dataset_param[im] = {'imgspec': selfbleu[im]}

    print(split, count_aligned)
    original_data[split] = dataset_param

print('data lengths', len(original_data['train']), len(original_data['val']), len(original_data['test']), '\n')

train_img_ids = set(original_data['train'].keys())

criterion = nn.MSELoss(reduction='none')

splits = ['val', 'test']

img_count = 0
k = len(original_data['train'])

for spl in splits:

    total_loss = 0.0
    img_count = 0
    targets = []
    outputs = []

    for img in original_data[spl]:

        img_count += 1
        # print('img count', img_count)

        target_ons = torch.tensor(original_data[spl][img]['imgspec'])
        samples_closest = retrieve_closest(img, train_img_ids, k_count=k)

        preds = 0.0
        normalizer = 0.0

        for sc in samples_closest:
            sc_img, sc_sim = sc

            # closest ones always retrieved from train
            img_spec = original_data['train'][sc_img]['imgspec']

            preds += img_spec * sc_sim
            normalizer += sc_sim

        final_pred = torch.tensor(preds / normalizer)
        targets.append(target_ons.item())
        outputs.append(final_pred.item())
        loss_mean = criterion(final_pred, target_ons)
        loss_mean_sqrt = math.sqrt(loss_mean.item())
        total_loss += loss_mean_sqrt

    print(img_count)

    # print(spl, 'k', k, 'loss sqrt total', total_loss)

    print(spl, 'k', k, 'ONSET interpretable loss avg sqrt', round(total_loss / len(original_data[spl]), 4))

    corr, pvalue = spearmanr(targets, outputs)
    print(spl, 'correlation target img spec vs. predicted img spec')
    print(round(corr, 4), round(pvalue, 4))

    print()
