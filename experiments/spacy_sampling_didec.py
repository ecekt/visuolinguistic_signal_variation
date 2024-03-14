import spacy
from spacy.lang.nl.examples import sentences
import os
import numpy as np
from collections import defaultdict

np.random.seed(42)

# spacy models of different sizes
nlp_sm = spacy.load("nl_core_news_sm")
nlp_md = spacy.load("nl_core_news_md")
nlp_lg = spacy.load("nl_core_news_lg")

count_f = 0

root_path = '/home/ece/Desktop/phd_2022/gaze_code/clipcap_dutch/data/didec/grammars/'

file_list = []

for root, subdirs, files in os.walk(root_path):

    for f in files:

        count_f += 1

        file_list.append(f)

print(count_f)

files = np.random.choice(file_list, 50, replace=False)

count = 0

for f in files:

    count += 1
    _, _, ppn, img = f.split('.')[0].split('_')

    grammar_file = os.path.join(root_path, f)

    with open(grammar_file, 'r') as g:
        lines = g.readlines()
        for line in lines:
            if 'public <s> = ' in line:
                utterance = line.split('public <s> = ')[1].split(';')[0]

    print('\nspacy small ' + str(count) + ' \n')
    doc = nlp_sm(utterance)
    print(doc.text)
    print()
    for token in doc:
        print(token.lemma_, token.pos_)

count = 0

for f in files:

    count += 1
    _, _, ppn, img = f.split('.')[0].split('_')

    grammar_file = os.path.join(root_path, f)

    with open(grammar_file, 'r') as g:
        lines = g.readlines()
        for line in lines:
            if 'public <s> = ' in line:
                utterance = line.split('public <s> = ')[1].split(';')[0]

    print('\nspacy medium ' + str(count) + ' \n')
    doc = nlp_md(utterance)
    print(doc.text)
    print()
    for token in doc:
        print(token.lemma_, token.pos_)

count = 0

for f in files:

    count += 1
    _, _, ppn, img = f.split('.')[0].split('_')

    grammar_file = os.path.join(root_path, f)

    with open(grammar_file, 'r') as g:
        lines = g.readlines()
        for line in lines:
            if 'public <s> = ' in line:
                utterance = line.split('public <s> = ')[1].split(';')[0]

    print('\nspacy large ' + str(count) + ' \n')
    doc = nlp_lg(utterance)
    print(doc.text)
    print()
    for token in doc:
        print(token.lemma_, token.pos_)
