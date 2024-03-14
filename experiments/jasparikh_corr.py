from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

prompts = defaultdict(list)

from nlgeval import NLGEval

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

jasparikh_humimgspec = []
# human scores
with open('jasparikh_imgspec_scores.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        jasparikh_humimgspec.append(float(line))

count = 0
count_bleuzero = 0

img_specificity = []

# filter out BLEU2 0.0
img_specificity_filtered = []
jasparikh_humimgspec_filtered = []

with open('jasparikh_imgspec_sents.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:

        if count % 100 == 0:
            print(count)
        count += 1

        sentences = line.split('\t')
        sentences = [s.lower().strip() for s in sentences]
        count_sents = len(sentences)
        assert count_sents == 5

        img_bleus = []

        for s in range(count_sents):
            hypo = sentences[s]

            set_others = set(range(count_sents)) - {s}
            refs = [sentences[j] for j in set_others]

            metrics_dict = nlgeval.compute_individual_metrics(refs, hypo)

            bleu = metrics_dict['Bleu_2']
            img_bleus.append(bleu)

        img_spec = np.mean(img_bleus)

        if round(img_spec, 3) == 0.0:
            count_bleuzero += 1
            print(sentences)
        else:
            img_specificity_filtered.append(img_spec)
            jasparikh_humimgspec_filtered.append(jasparikh_humimgspec[count - 1])

        print(count, round(img_spec, 3))

        img_specificity.append(img_spec)

corr, pvalue = spearmanr(img_specificity, jasparikh_humimgspec)
print('correlation BLEU-2 imgspec vs. Jas-Parikh Human imgspec')
print(round(corr, 3), round(pvalue, 3))

print('zero bleu2 count', count_bleuzero, '\n')  # 103

corr, pvalue = spearmanr(img_specificity_filtered, jasparikh_humimgspec_filtered)
print('correlation FILTERED BLEU-2 imgspec vs. Jas-Parikh Human imgspec')
print(round(corr, 3), round(pvalue, 3))
