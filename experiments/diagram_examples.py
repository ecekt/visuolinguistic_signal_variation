import os
import json

count_f = 0

img_id = '713328'
print(img_id)

for root, subdirs, files in os.walk('data/alignments_whisperX'):

    for f in files:

        count_f += 1

        _, _, ppn, img = f.split('.')[0].split('_')

        if img == img_id:

            with open(os.path.join(root, f)) as jj:
                alg = json.load(jj)
                text = [w['text'] for w in alg]
                text_str = ' '.join(text)
                print(text_str)

# print(count_f)

with open('data/didec/didec_image_specificity_selfbleu.json', 'r') as f:
    selfbleu = json.load(f)

with open('startingpoint_var.json', 'r') as f:
    spvar = json.load(f)

with open('gaze_variation_IOU.json', 'r') as f:
    gazeIOU = json.load(f)

with open('onset_means.json', 'r') as f:
    onsets = json.load(f)

print()
print('Img spec BLEU-2:', selfbleu[img_id])
print('SP variation:', spvar[img_id])
print('Mean onset:', onsets[img_id])
print('Gaze variation', gazeIOU[img_id])
