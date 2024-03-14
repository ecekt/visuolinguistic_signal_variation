import json
import math
import pickle
from collections import defaultdict
import numpy as np
from torch import nn
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

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
            onset_list.append(delay)

        dataset_param[im] = {'mean_ons': np.mean(onset_list),
                             'std_ons': np.std(onset_list)}

    print(split, count_aligned)
    original_data[split] = dataset_param

print('data lengths', len(original_data['train']), len(original_data['val']), len(original_data['test']), '\n')

train_img_ids = set(original_data['train'].keys())

criterion = nn.MSELoss(reduction='none')

splits = ['train', 'val', 'test']

for spl in splits:

    total_loss = 0.0
    img_count = 0
    means = []
    stds = []
    specs = []

    for img in original_data[spl]:

        specs.append(selfbleu[img])

        img_count += 1
        # print('img count', img_count)

        target_mean = original_data[spl][img]['mean_ons']
        means.append(target_mean)

        target_std = original_data[spl][img]['std_ons']
        stds.append(target_std)

    plt.hist(means)
    plt.title(spl + ' mean onsets per image')
    plt.xlabel('Onset avg in sec')
    plt.ylabel('Frequency')
    plt.xlim(0, 10)
    plt.savefig('plot_onset_means_' + spl + '.png')
    plt.close()

    plt.hist(stds)
    plt.title(spl + ' std onsets per image')
    plt.xlabel('Onset std')
    plt.ylabel('Frequency')
    plt.xlim(0, 10)
    plt.savefig('plot_onset_stds_' + spl + '.png')
    plt.close()

    plt.hist(specs)
    plt.title(spl + ' image specificity based on Self-BLEU-2')
    plt.xlabel('Image specificity')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.savefig('plot_onset_imgspec_bleu2s_' + spl + '.png')
    plt.close()

    plt.scatter(means, specs)
    a, b = np.polyfit(means, specs, 1)
    plt.plot(means, a * np.array(means) + b, color='black')
    plt.title(spl + ' mean onset vs. image specificity based on Self-BLEU-2')
    plt.ylim(0, 1)
    plt.xlabel('Mean onset')
    plt.ylabel('Image specificity')
    plt.savefig('plot_corr_meanonset_imgspec_bleu2s_' + spl + '.png')
    plt.close()

    plt.scatter(stds, specs)
    a, b = np.polyfit(stds, specs, 1)
    plt.plot(stds, a * np.array(stds) + b, color='black')
    plt.title(spl + ' std onset vs. image specificity based on Self-BLEU-2')
    plt.ylim(0, 1)
    plt.xlabel('STD onset')
    plt.ylabel('Image specificity')
    plt.savefig('plot_corr_stdonset_imgspec_bleu2s_' + spl + '.png')
    plt.close()

