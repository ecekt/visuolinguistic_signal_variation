import json
import random
from math import floor
from collections import defaultdict
import os

random.seed(42)

if not os.path.isdir('data/didec/random_splits'):
    os.mkdir('data/didec/random_splits')


trial_count = 50

train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

datapaths = {'train': 'data/didec/split_train.json',
             'val': 'data/didec/split_val.json',
             'test': 'data/didec/split_test.json'}

img_list = []
full_data = dict()

# start with the orig img_list
for split in datapaths:
    print(split)
    path = datapaths[split]

    with open(path, 'r') as f:
        datasplit = json.load(f)

    img_list += list(datasplit.keys())
    full_data.update(datasplit)

# create splits
for c in range(trial_count):

    print('random c', c)

    # shuffle randomly and take percentages into account
    random.shuffle(img_list)

    len_list = len(img_list)
    print(len_list)

    print(str(len_list * 0.8), str(len_list * 0.1))

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
        train_set[i] = full_data[i]

    for i in val_imgs:
        val_set[i] = full_data[i]

    for i in test_imgs:
        test_set[i] = full_data[i]

    print('set sizes in terms of images')
    print(len(train_set), len(val_set), len(test_set))


    #CREATE THE JSON FILES

    with open('data/didec/random_splits/split_train_' + str(c) + '.json', 'w') as file:
        json.dump(train_set, file)

    with open('data/didec/random_splits/split_val_' + str(c) + '.json', 'w') as file:
        json.dump(val_set, file)

    with open('data/didec/random_splits/split_test_' + str(c) + '.json', 'w') as file:
        json.dump(test_set, file)
