import os
import numpy as np
from collections import defaultdict

imgs_per_ppn = defaultdict(list)
ppns_per_img = defaultdict(list)

count = 0

for root, dirs, files in os.walk('data/alignments_whisperX/'):

    for f in files:
        _, _, ppn, im = f.split('.')[0].split('_')

        if im not in imgs_per_ppn[ppn]:
            imgs_per_ppn[ppn].append(im)

        if ppn not in ppns_per_img:
            ppns_per_img[im].append(ppn)

        count += 1

avg_imgs_per_ppn = []
avg_ppns_per_img = []

for ppn in imgs_per_ppn:
    avg_imgs_per_ppn.append(len(imgs_per_ppn[ppn]))

for im in ppns_per_img:
    avg_ppns_per_img.append(len(ppns_per_img[im]))

avg_imgs_per_ppn = np.mean(avg_imgs_per_ppn)
avg_ppns_per_img = np.mean(avg_ppns_per_img)

print(round(avg_imgs_per_ppn, 3))
print(round(avg_ppns_per_img, 3))
print(count)
