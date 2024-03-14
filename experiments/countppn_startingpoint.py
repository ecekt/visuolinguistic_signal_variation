import csv
import json
import math
import pickle
from collections import defaultdict
import numpy as np
import torch
import argparse
import os
from collections import Counter

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, GPT2Model, GPT2Config, set_seed, get_scheduler

from tqdm.auto import tqdm
import datetime

import spacy
from spacy.lang.nl.examples import sentences

nlp = spacy.load("nl_core_news_lg")
# doc = nlp(sentences[0])
# print(doc.text)
# for token in doc:
#     print(token.text, token.pos_, token.dep_, token.is_stop)
#
# stopwords = nlp.Defaults.stop_words

if __name__ == '__main__':

    datapaths = {'train': 'data/didec/split_train.json',
                 'val': 'data/didec/split_val.json',
                 'test': 'data/didec/split_test.json'}

    vocab = []
    im_set = defaultdict(set)

    for split in datapaths:

        print(split)
        path = datapaths[split]

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
                                    break
                                else:
                                    print('foto or aantal or uh mentioned')

                        if content_start_word == '':
                            print('no nouns here')
                            content_start_word = '<unk>'

                        im_set[im].add(content_start_word)

print(len(im_set))

lens = [len(im_set[img]) for img in im_set]
print(round(np.mean(lens), 4))
