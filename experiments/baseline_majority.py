import json
from collections import defaultdict
import numpy as np
import torch
from collections import Counter

import spacy
from spacy.lang.nl.examples import sentences

nlp = spacy.load("nl_core_news_lg")
# doc = nlp(sentences[0])
# print(doc.text)
# for token in doc:
#     print(token.text, token.pos_, token.dep_, token.is_stop)
#
# stopwords = nlp.Defaults.stop_words

datapaths = {'train': 'data/didec/split_train.json',
             'val': 'data/didec/split_val.json',
             'test': 'data/didec/split_test.json'}

vocab = []
vocab4imgs = defaultdict(set)
vocab_spcounter_imgs = defaultdict(Counter)

original_data = dict()

for split in datapaths:

    print(split)
    path = datapaths[split]

    dataset_param = defaultdict(dict)

    with open(path, 'r') as f:
        datasplit = json.load(f)

    count_aligned = 0

    for im in datasplit:
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

                            if content_start_word not in ['aantal', 'uh', 'paar'] and 'foto' not in content_start_word:
                                vocab_spcounter_imgs[im].update([content_start_word])
                                break
                            else:
                                print('foto or aantal or uh or paar mentioned')

                    for token in doc:
                        if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                            if token.text.lower() not in ['aantal', 'uh', 'paar'] and 'foto' not in token.text.lower():
                                vocab4imgs[im].add(token.lemma_.lower())

                    if content_start_word == '':
                        print('no nouns here')
                        content_start_word = '<unk>'
                        vocab4imgs[im].add(content_start_word)
                        vocab_spcounter_imgs[im].update([content_start_word])

            vocab.append(content_start_word)

            dataset_param[im][ppn] = {'text': text_str,
                                      'start': content_start_word}

    print(split, count_aligned)
    original_data[split] = dataset_param

splits = ['train', 'val', 'test']

print('\npredict majority for each caption per image')
for spl in splits:
    # masked baseline for EACH CAPTION
    acc = 0.0
    counter = 0.0
    for img in original_data[spl]:
        for ppn in original_data[spl][img]:

            target_start = original_data[spl][img][ppn]['start']
            predict = vocab_spcounter_imgs[img].most_common(1)[0][0]

            counter += 1

            if predict == target_start:
                acc += 1

    split_acc = acc / counter
    print(spl, round(np.mean(split_acc), 4))
