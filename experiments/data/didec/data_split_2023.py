import numpy as np
import json
import random
import os
from math import floor
from collections import defaultdict

# use whisperX alignments
# which has all audio files aligned with text
# as opposed to CMU Sphinx, which output no alignments for 38 samples

train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

count_f = 0

img_ppn_list = defaultdict(list)

for root, subdirs, files in os.walk('../alignments_whisperX'):

    for f in files:

        count_f += 1
        if count_f % 10 == 0:
            print(count_f)

        _, _, ppn, img = f.split('.')[0].split('_')

        img_ppn_list[img].append(ppn)

print('total', count_f)

img_list = list(img_ppn_list.keys())
len_list = len(img_list)
print(len_list, str(len_list * 0.8), str(len_list * 0.1))

#shuffle randomly and take percentages into account
random.shuffle(img_list)

train_ind = round(len_list * train_percentage) + 1
val_ind = train_ind + floor(len_list * val_percentage)
test_ind = val_ind + floor(len_list * test_percentage)

print(train_ind, val_ind, test_ind)

train_imgs = img_list[:train_ind]
val_imgs = img_list[train_ind:val_ind]
test_imgs = img_list[val_ind:]

print('image counts')
print(len(train_imgs), len(val_imgs), len(test_imgs))

train_set = defaultdict(list)
val_set = defaultdict(list)
test_set = defaultdict(list)

for i in train_imgs:
    train_set[i] = img_ppn_list[i]

for i in val_imgs:
    val_set[i] = img_ppn_list[i]

for i in test_imgs:
    test_set[i] = img_ppn_list[i]

with open('split_train.json', 'w') as file:
    json.dump(train_set, file)

with open('split_val.json', 'w') as file:
    json.dump(val_set, file)

with open('split_test.json', 'w') as file:
    json.dump(test_set, file)

