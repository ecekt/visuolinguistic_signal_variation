import pickle
import numpy as np
import json
from collections import defaultdict
import math
import torch
from torch import nn
from scipy.stats import normaltest, ttest_ind
import os
import matplotlib.pyplot as plt

list_ppns = defaultdict(list)
all_ppns = set()
ppn2list = defaultdict(str)

for root, subdirs, files in os.walk('/home/ece/Downloads/DIDEC_only_the_eyetracking_data/'
                                    'DIDEC_only_the_eyetracking_data/Image_description'):

    for dir in subdirs:
        if 'pp' in dir:
            #print(root, dir)
            ppn = str(int(dir.split('pp')[1]))
            all_ppns.add(ppn)

            if 'List1' in root:
                list_ppns['1'].append(ppn)
                ppn2list[ppn] = '1'
            elif 'List2' in root:
                list_ppns['2'].append(ppn)
                ppn2list[ppn] = '2'
            elif 'List3' in root:
                list_ppns['3'].append(ppn)
                ppn2list[ppn] = '3'

assert len(all_ppns) == len([p for i in list_ppns for p in list_ppns[i]])

with open('data/didec/clip_preprocessed_didec_fx2SAMcropped_B32_cpu.pickle', 'rb') as f:
    clip_preprocessed_fixations = pickle.load(f)

with open('onset_means.json', 'r') as f:
    onsets = json.load(f)

path_alignments = 'data/alignments_whisperX/'
list_speech_onsets = defaultdict(list)
ppn_speech_onsets = defaultdict(list)
list_imgs = defaultdict(set)

for pair in clip_preprocessed_fixations:
    ppn, im = pair

    alignment_file = path_alignments + 'aligned_whisperx_' + str(ppn) + '_' + str(im) + '.json'

    with open(alignment_file, 'r') as f:
        alg = json.load(f)

    onset = alg[0]['start']
    ppn_speech_onsets[ppn].append(onset)

    list_of_ppn = ppn2list[ppn]

    list_speech_onsets[list_of_ppn].append(onset)
    list_imgs[list_of_ppn].add(im)

list_img_meanons = defaultdict(list)
for lop in list_imgs:
    for imgg in list_imgs[lop]:
        list_img_meanons[lop].append(onsets[imgg])

means1 = list_img_meanons['1']
means2 = list_img_meanons['2']
means3 = list_img_meanons['3']
tstat, pvalue = ttest_ind(means1, means2)
print('t-test MEAN onsets list1 vs. list2')
print(round(tstat, 4), round(pvalue, 4))
tstat, pvalue = ttest_ind(means1, means3)
print('t-test MEAN onsets list1 vs. list3')
print(round(tstat, 4), round(pvalue, 4))
tstat, pvalue = ttest_ind(means2, means3)
print('t-test MEAN onsets list2 vs. list3')
print(round(tstat, 4), round(pvalue, 4))


# t-tests between lists
onsets_list1 = list_speech_onsets['1']
onsets_list2 = list_speech_onsets['2']
onsets_list3 = list_speech_onsets['3']

tstat, pvalue = ttest_ind(onsets_list1, onsets_list2)
print(ppn, 't-test onsets list1 vs. list2')
print(round(tstat, 4), round(pvalue, 4))

tstat, pvalue = ttest_ind(onsets_list1, onsets_list3)
print(ppn, 't-test onsets list1 vs. list3')
print(round(tstat, 4), round(pvalue, 4))

tstat, pvalue = ttest_ind(onsets_list2, onsets_list3)
print(ppn, 't-test onsets list2 vs. list3')
print(round(tstat, 4), round(pvalue, 4))

print(round(np.mean(list_speech_onsets['1']), 4))
print(round(np.mean(list_speech_onsets['2']), 4))
print(round(np.mean(list_speech_onsets['3']), 4))

# print('ppns')
# for p in ppn_speech_onsets:
#     print(round(np.mean(ppn_speech_onsets[p]), 4))
#
# for ll in list_ppns:
#     subset_onsets = []
#     subset_ppns = list_ppns[ll]
#     for sp in subset_ppns:
#         subset_onsets.extend(ppn_speech_onsets[sp])
#
#     plt.hist(subset_onsets)
#     plt.title('ppn onsets of list' + ll)
#     plt.savefig('hist_list' + ll + '.png')
#     plt.close()
#
# for ll in list_ppns:
#     print('list no ' + ll)
#     subset_ppns = list_ppns[ll]
#     for sp in subset_ppns:
#         others = set(subset_ppns) - {sp}
#
#         for op in others:
#             tstat, pvalue = ttest_ind(ppn_speech_onsets[sp], ppn_speech_onsets[op])
#             print(ppn, 't-test onsets ' + sp + ' vs. ' + op)
#             print(round(tstat, 4), round(pvalue, 4))
