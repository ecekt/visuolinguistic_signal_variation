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

c = 0
datapaths = {'train': 'data/didec/random_splits/split_train_' + str(c) + '.json',
             'val': 'data/didec/random_splits/split_val_' + str(c) + '.json',
             'test': 'data/didec/random_splits/split_test_' + str(c) + '.json'}

original_data = dict()

vocab = set()
sp_counts = dict()

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
        sp_counts[im] = len(set(im_sps))
        dataset_param[im] = {'start': im_start_word}

    print(split, count_aligned)
    original_data[split] = dataset_param

print('data lengths', len(original_data['train']), len(original_data['val']), len(original_data['test']), '\n')

sp_avg = np.mean([sp_counts[im] for im in sp_counts])
print(sp_avg)

with open('startingpoint_var.json', 'w') as f:
    json.dump(sp_counts, f)