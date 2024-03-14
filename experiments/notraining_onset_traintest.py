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

trial_count = 50

coefs_targetpred = []
p_targetpred = []

coefs_speconset = []
p_speconset = []

pred_ranges = []
pred_losses = []

target_ranges = []

predictions_across_splits = defaultdict(list)

for c in range(trial_count):
    datapaths = {'train': 'data/didec/random_splits/split_train_' + str(c) + '.json',
                 'val': 'data/didec/random_splits/split_val_' + str(c) + '.json',
                 'test': 'data/didec/random_splits/split_test_' + str(c) + '.json'}

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
                        # print(count_aligned)

                        for line in lines:
                            if count_line == 0:
                                delay = line['start']

                            count_line += 1

                # print('speech onset', delay, '\n')
                onset_list.append(delay)

            dataset_param[im] = {'mean_ons': np.mean(onset_list)}

        print(split, count_aligned)
        original_data[split] = dataset_param

    print('data lengths', len(original_data['train']), len(original_data['val']), len(original_data['test']), '\n')

    # combine train + val
    original_data['train'].update(original_data['val'])
    del original_data['val']

    train_img_ids = set(original_data['train'].keys())

    criterion = nn.MSELoss(reduction='none')

    # only check test, because we combined train + val
    splits = ['test']

    img_count = 0
    k = len(original_data['train'])

    for spl in splits:

        total_loss = 0.0
        img_count = 0
        targets = []
        outputs = []
        specs = []

        for img in original_data[spl]:

            specs.append(selfbleu[img])

            img_count += 1
            # print('img count', img_count)

            target_ons = torch.tensor(original_data[spl][img]['mean_ons'])
            samples_closest = retrieve_closest(img, train_img_ids, k_count=k)

            preds = 0.0
            normalizer = 0.0

            for sc in samples_closest:
                sc_img, sc_sim = sc

                # closest ones always retrieved from train
                mean_ons = original_data['train'][sc_img]['mean_ons']

                preds += mean_ons * sc_sim
                normalizer += sc_sim

            final_pred = torch.tensor(preds / normalizer)
            targets.append(target_ons.item())
            outputs.append(final_pred.item())
            predictions_across_splits[img].append(final_pred.item())
            loss_mean = criterion(final_pred, target_ons)
            loss_mean_sqrt = math.sqrt(loss_mean.item())
            total_loss += loss_mean_sqrt

        print(img_count)

        # print(spl, 'k', k, 'loss sqrt total', total_loss)

        print(spl, 'k', k, 'ONSET interpretable loss avg sqrt', round(total_loss / len(original_data[spl]), 4))

        pred_losses.append(total_loss / len(original_data[spl]))
        pred_ranges.append((min(outputs), max(outputs)))
        target_ranges.append((min(targets), max(targets)))

        corr, pvalue = spearmanr(targets, outputs)
        print(spl, 'correlation avg target onset vs. predicted onset')
        print(round(corr, 4), round(pvalue, 4))

        coefs_targetpred.append(corr)
        p_targetpred.append(pvalue)

        corr, pvalue = spearmanr(specs, targets)
        print(spl, 'correlation image spec bleu-2 vs. avg target onset')
        print(round(corr, 4), round(pvalue, 4))

        coefs_speconset.append(corr)
        p_speconset.append(pvalue)

        print()

    # img_count = 0

    # for spl in ['train']:
    #
    #     img_count = 0
    #     targets = []
    #     specs = []
    #
    #     for img in original_data[spl]:
    #
    #         specs.append(selfbleu[img])
    #
    #         img_count += 1
    #         # print('img count', img_count)
    #
    #         target_ons = torch.tensor(original_data[spl][img]['mean_ons'])
    #
    #         targets.append(target_ons.item())
    #
    #     print(img_count)
    #
    #     corr, pvalue = spearmanr(specs, targets)
    #     print(spl, 'correlation image spec bleu-2 vs. avg target onset')
    #     print(round(corr, 4), round(pvalue, 4))
print()

count_moderate = 0
count_large = 0
count_sig = 0

for i in range(len(p_targetpred)):

    if p_targetpred[i] < 0.05:
        # significant
        count_sig += 1

        if 0.3 <= abs(coefs_targetpred[i]) < 0.6:
            count_moderate += 1
        elif abs(coefs_targetpred[i]) >= 0.6:
            count_large += 1

print('corr between target onset and predicted onset')
print(len(p_targetpred))
print(count_sig, count_moderate, count_large)

count_moderate = 0
count_large = 0
count_sig = 0

for i in range(len(p_speconset)):

    if p_speconset[i] < 0.05:
        # significant
        count_sig += 1

        if 0.3 <= abs(coefs_speconset[i]) < 0.6:
            count_moderate += 1
        elif abs(coefs_speconset[i]) >= 0.6:
            count_large += 1

print('corr between image specificity and target onset')
print(len(p_speconset))
print(count_sig, count_moderate, count_large)

print('loss avg over runs', round(np.mean(pred_losses), 4))
print('coef avg over runs', round(np.mean(coefs_targetpred), 4))

mins = []
maxs = []

for i in range(len(pred_ranges)):

    mins.append(pred_ranges[i][0])
    maxs.append(pred_ranges[i][1])

print('avg min', round(np.mean(mins), 4))
print('avg max', round(np.mean(maxs), 4))
print('min min', round(np.min(mins), 4))
print('max max', round(np.max(maxs), 4))


print('target ranges below')
mins = []
maxs = []

for i in range(len(target_ranges)):

    mins.append(target_ranges[i][0])
    maxs.append(target_ranges[i][1])

print('avg min', round(np.mean(mins), 4))
print('avg max', round(np.mean(maxs), 4))
print('min min', round(np.min(mins), 4))
print('max max', round(np.max(maxs), 4))

min_pred_im = ''
max_pred_im = ''
min_pred_val = math.inf
max_pred_val = -math.inf

for img in predictions_across_splits:
    img_pred_mean = np.mean(predictions_across_splits[img])

    if img_pred_mean < min_pred_val:
        min_pred_val = img_pred_mean
        min_pred_im = img

    if img_pred_mean > max_pred_val:
        max_pred_val = img_pred_mean
        max_pred_im = img

print('min pred mean onset:', min_pred_im, str(round(min_pred_val, 3)))
print('max pred mean onset:', max_pred_im, str(round(max_pred_val, 3)))
