import json
import os
from collections import defaultdict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

prompts = defaultdict(list)

from nlgeval import NLGEval
nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
# test
# hypo = 'this is a test sentence'
# refs = ['this is a sentence', 'this is a test'] # this is a book
#
# metrics_dict = nlgeval.compute_individual_metrics(refs, hypo)
# print(metrics_dict)
# print(sentence_bleu([rf.split() for rf in refs], hypo.split()))

for root, dirs, files in os.walk('data/alignments_whisperX/'):

    for f in files:
        alignment_file = os.path.join(root, f)
        _, _, ppn, im = f.split('.')[0].split('_')

        with open(alignment_file, 'r') as j:
            lines = json.load(j)
            text = []

            if len(lines) == 0:
                print('empty', f)
            else:
                for line in lines:
                    text.append(line["text"])

            prompt = ' '.join(text)
            prompts[im].append(prompt)

count = 0

img_specificity = defaultdict()

for im in prompts:

    sentences = prompts[im]
    count_sents = len(sentences)

    img_bleus = []

    if count % 100 == 0:
        print(count)
    count += 1

    for s in range(count_sents):
        hypo = sentences[s]

        set_others = set(range(count_sents)) - {s}
        refs = [sentences[j] for j in set_others]

        metrics_dict = nlgeval.compute_individual_metrics(refs, hypo)
        bleu = metrics_dict['Bleu_2']
        img_bleus.append(bleu)

    img_spec = np.mean(img_bleus)

    print(im, round(img_spec, 3))

    img_specificity[im] = img_spec

with open('data/didec/didec_image_specificity_selfbleu.json', 'w') as f:
    json.dump(img_specificity, f)
