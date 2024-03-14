import json
import os
from collections import defaultdict

with open('split_train_old.json', 'r') as file:
    train_set = json.load(file)

with open('split_val_old.json', 'r') as file:
    val_set = json.load(file)

with open('split_test_old.json', 'r') as file:
    test_set = json.load(file)

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

for im in img_ppn_list:

    if im in train_set:
        train_set[im] = img_ppn_list[im]
    elif im in val_set:
        val_set[im] = img_ppn_list[im]
    elif im in test_set:
        test_set[im] = img_ppn_list[im]

with open('split_train.json', 'w') as file:
    json.dump(train_set, file)

with open('split_val.json', 'w') as file:
    json.dump(val_set, file)

with open('split_test.json', 'w') as file:
    json.dump(test_set, file)

