import torch
import json
import os
from collections import defaultdict

if not os.path.isdir('data/didec/bertdutch_sentencereps_CLS'):
    os.mkdir('data/didec/bertdutch_sentencereps_CLS')

from transformers import AutoTokenizer, AutoModel, set_seed

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")

prompts = defaultdict()

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
            prompts[(ppn, im)] = prompt

count = 0
for pair in prompts:

    if count % 100 == 0:
        print(count)
    count += 1

    input_text = prompts[pair]
    p, i = pair
    encoded_input = tokenizer(input_text, return_tensors='pt')
    output = model(**encoded_input)
    sentence_rep = output.last_hidden_state[0][0]

    torch.save(sentence_rep, 'data/didec/bertdutch_sentencereps_CLS/bertrep_CLS_' + p + '_' + i + '.pt')

