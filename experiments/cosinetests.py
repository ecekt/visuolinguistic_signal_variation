import torch
from collections import defaultdict
import pickle
from scipy.spatial import distance

# with open('clip_im_sims.pickle', 'rb') as f:
#     sims = pickle.load(f)

with open('data/didec/clip_preprocessed_didec_images_cpu.pickle', 'rb') as f:
    clip_preprocessed_images = pickle.load(f)

sims = defaultdict(dict)
count = 0

for im in clip_preprocessed_images:
    others = set(clip_preprocessed_images.keys()) - {im}

    for o in others:
        sim = 1 - distance.cosine(clip_preprocessed_images[im][0], clip_preprocessed_images[o][0])
        sims[im][o] = sim

        count += 1

        if count % 1000 == 0:
            print(count)

print(count)


with open('clip_im_sims.pickle', 'wb') as f:
    pickle.dump(sims, f)


