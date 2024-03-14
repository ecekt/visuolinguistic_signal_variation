import json
import math
import pickle
from collections import defaultdict, Counter
import numpy as np
from torch import nn
import torch
from scipy.stats import spearmanr
import spacy
from spacy.lang.nl.examples import sentences

nlp = spacy.load("nl_core_news_lg")
# doc = nlp(sentences[0])
# print(doc.text)
# for token in doc:
#     print(token.text, token.pos_, token.dep_, token.is_stop)
#
# stopwords = nlp.Defaults.stop_words

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

pred_accs = []

for c in range(trial_count):
    datapaths = {'train': 'data/didec/random_splits/split_train_' + str(c) + '.json',
                 'val': 'data/didec/random_splits/split_val_' + str(c) + '.json',
                 'test': 'data/didec/random_splits/split_test_' + str(c) + '.json'}

    original_data = dict()

    vocab = set()

    for split in datapaths:

        print(split)
        path = datapaths[split]

        dataset_param = defaultdict(dict)

        with open(path, 'r') as f:
            datasplit = json.load(f)

        count_aligned = 0

        for im in datasplit:

            onset_list = []
            im_sps = []

            for ppn in datasplit[im]:

                alignment_file = 'data/alignments_whisperX/aligned_whisperx_' + ppn + '_' + im + '.json'

                with open(alignment_file, 'r') as j:
                    lines = json.load(j)

                    text = []

                    if len(lines) == 0:
                        print('empty', f)
                    else:
                        count_aligned += 1
                        print(count_aligned)

                        for line in lines:
                            text.append(line['text'])

                        text_str = ' '.join(text)
                        print(text_str)

                        doc = nlp(text_str)

                        content_start_word = ''

                        for token in doc:
                            # print the first content word
                            if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                                print(token.text, token.pos_)
                                content_start_word = token.lemma_.lower()

                                if content_start_word not in ['aantal', 'uh',
                                                              'paar'] and 'foto' not in content_start_word:
                                    break
                                else:
                                    print('foto or aantal or uh mentioned')

                        if content_start_word == '':
                            print('no nouns here')
                            content_start_word = '<unk>'

                        im_sps.append(content_start_word)

                        vocab.add(content_start_word)

            im_start_word = Counter(im_sps).most_common(1)[0][0]
            dataset_param[im] = {'start': im_start_word}

        print(split, count_aligned)
        original_data[split] = dataset_param

    print('data lengths', len(original_data['train']), len(original_data['val']), len(original_data['test']), '\n')

    vocab = list(vocab)

    # combine train + val
    original_data['train'].update(original_data['val'])
    del original_data['val']

    train_img_ids = set(original_data['train'].keys())

    # only check test, because we combined train + val
    splits = ['test']

    img_count = 0
    k = 10 # len(original_data['train'])

    for spl in splits:

        total_acc = 0.0
        img_count = 0

        for img in original_data[spl]:

            img_count += 1
            # print('img count', img_count)

            target_word = original_data[spl][img]['start']
            target_index = vocab.index(target_word)
            target_ons = np.zeros(len(vocab))
            target_ons[target_index] = 1
            samples_closest = retrieve_closest(img, train_img_ids, k_count=k)

            preds = np.zeros(len(vocab))
            normalizer = 0.0

            for sc in samples_closest:
                sc_img, sc_sim = sc

                # closest ones always retrieved from train
                pred_word = original_data['train'][sc_img]['start']
                pred_index = vocab.index(pred_word)
                pred_ons = np.zeros(len(vocab))
                pred_ons[pred_index] = 1

                preds += pred_ons * sc_sim
                # normalizer += sc_sim

            final_ind = np.argmax(preds)
            if final_ind == target_index:
                total_acc += 1

        pred_acc = total_acc / img_count
        print(img_count, 'acc', str(round(pred_acc, 4)))
        pred_accs.append(pred_acc)

print('prediction accuracy avg over runs', round(np.mean(pred_accs), 4))
