import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math


datapaths = {'train': '../data/didec/split_train.json',
             'val': '../data/didec/split_val.json',
             'test': '../data/didec/split_test.json'}

for split in datapaths:

    print(split)
    path = datapaths[split]

    with open(path, 'r') as f:
        datasplit = json.load(f)

    count_aligned = 0
    speech_onsets = []
    image_onset_stats = defaultdict()

    for im in datasplit:

        image_onsets = []

        for ppn in datasplit[im]:

            alignment_file = '../data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + im + '.json'

            with open(alignment_file, 'r') as j:
                lines = json.load(j)

                onset = 0

                if len(lines) == 0:
                    print('empty', f)
                else:
                    count_aligned += 1
                    print(count_aligned)

                    for line in lines:
                        onset = line['start']
                        break

            print('speech onset', onset, '\n')

            speech_onsets.append(onset)
            image_onsets.append(onset)

        print('image onset stats for', im)
        print(round(np.mean(image_onsets), 3))
        print(round(np.std(image_onsets), 3))
        median = np.median(image_onsets)
        mean = np.mean(image_onsets)
        std = np.std(image_onsets)
        range = np.max(image_onsets) - np.min(image_onsets)

        image_onset_stats.update({im: {'mean': mean, 'std': std, 'range': range}})

        print('median', median)
        maxy = len(image_onsets) / 2 + 10
        plt.hist(image_onsets)
        plt.vlines(median, ymin=0, ymax=maxy, color='orange')
        plt.vlines(mean, ymin=0, ymax=maxy, color='red')

        plt.title(str(im))
        plt.xlabel('Onset in sec')
        plt.ylabel('Frequency')
        plt.savefig('../figures_whisperx/onset_' + im + '_whisperx.png')
        plt.close()

        print()

    with open(split + '_img_onset_stats_whisperx.json', 'w') as f:
        json.dump(image_onset_stats, f)

    print('dumped', split, count_aligned, len(image_onset_stats))

    print('split speech onset', np.mean(speech_onsets), np.min(speech_onsets), np.max(speech_onsets))
    median = np.median(speech_onsets)
    mean = np.mean(speech_onsets)

    print('median', median)
    maxy = len(speech_onsets) / 2 + 10

    plt.hist(speech_onsets)
    plt.vlines(median, ymin=0, ymax=maxy, color='orange')
    plt.vlines(mean, ymin=0, ymax=maxy, color='red')

    plt.title(split.upper())
    plt.xlabel('Onset in sec')
    plt.ylabel('Frequency')
    plt.savefig('../figures_whisperx/onset_' + split + '_whisperx.png')
    plt.close()
