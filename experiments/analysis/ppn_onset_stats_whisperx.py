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
    ppn_onset_stats = defaultdict()

    for im in datasplit:

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

            if ppn not in ppn_onset_stats:
                ppn_onset_stats.update({ppn: [onset]})
            else:
                ppn_onset_stats[ppn].append(onset)

    print('speech onsets mean', np.mean(speech_onsets))
    print(count_aligned)

    means_per_ppn = [np.mean(ppn_onset_stats[ppn]) for ppn in ppn_onset_stats]

    plt.hist(means_per_ppn)

    plt.title(split.upper())
    plt.xlabel('Mean onset per ppn in sec')
    plt.ylabel('Frequency')
    plt.savefig('onset_meanperppn_' + split + '_whisperx.png')
    plt.close()
