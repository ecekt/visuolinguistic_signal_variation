import os
import torch
import json
from collections import defaultdict
import pickle

bbox_preprocessed_fxs = defaultdict(dict)

# dtype on cuda float16, on cpu float32
device = "cpu"

count_bbox = 0

for root, dirs, files in os.walk('/home/ece/Desktop/phd_2022/gaze_code/segment/fx2SAM_cropped_BBOXCOORDS'):

    for f in files:

        if '.json' in f:
            count_bbox += 1
            if count_bbox % 10 == 0:
                print(count_bbox)

            _, ppn, img_id, fx_id = f.split('.')[0].split('_')

            with open(os.path.join(root, f), 'r') as f:
                bbox = json.load(f)
                bbox_preprocessed_fxs[(ppn, img_id)][fx_id] = bbox

print(count_bbox, len(bbox_preprocessed_fxs))

print('saving bbox_preprocessed_didec_fx2SAMcropped_B32_cpu')
with open('data/didec/bbox_preprocessed_didec_fx2SAMcropped_B32_cpu.pickle', 'wb') as f:
    pickle.dump(bbox_preprocessed_fxs, f)
