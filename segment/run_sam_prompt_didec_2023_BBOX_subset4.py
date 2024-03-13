from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
sam = sam_model_registry["vit_l"](checkpoint="models/sam_vit_l_0b3195.pth")
sam.to(device=device)

predictor = SamPredictor(sam)

with open('didec/fixation_events_DS_2023.json', 'r') as f:
    fixations = json.load(f)

if not os.path.exists('fx2SAM_BBOXCOORDS'):
    os.mkdir('fx2SAM_BBOXCOORDS')

if not os.path.exists('fx2SAM_cropped_BBOXCOORDS'):
    os.mkdir('fx2SAM_cropped_BBOXCOORDS')

count_sample = 0

ppn_subset = ['26', '16', '24', '10', '46', '48', '36', '8', '23', '35', '105', '104']

for ppn in ppn_subset:
    for im in fixations[ppn]:

        print(count_sample)
        count_sample += 1

        image = cv2.imread('images/' + im + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fx_no = 0

        for fx_window in fixations[ppn][im]:

            input_point = []
            input_label = []

            for fx in fx_window:
                _, _, _, xp, yp = fx
                xp = float(xp) - 206
                yp = float(yp) - 50

                input_point.append([xp, yp])
                input_label.append(1)

            input_point = np.array(input_point)
            input_label = np.array(input_label)

            predictor.set_image(image)

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            # selected_index = np.argmax(scores)
            # mask = masks[selected_index]
            # score = scores[selected_index]

            # multimask_output False
            mask = masks[0]
            score = scores[0]

            mask_i = [i for i in range(len(mask)) for j in range(len(mask[i])) if mask[i][j]]
            mask_j = [j for i in range(len(mask)) for j in range(len(mask[i])) if mask[i][j]]

            min_y = min(mask_i)
            max_y = max(mask_i)

            min_x = min(mask_j)
            max_x = max(mask_j)

            # plt.figure(figsize=(10, 10))
            # plt.imshow(image)
            # show_mask(mask, plt.gca())
            # show_points(input_point, input_label, plt.gca())
            # show_box([min_x, min_y, max_x, max_y], plt.gca())
            # plt.title(f"Mask score: {score:.3f}", fontsize=18)
            # plt.axis('off')
            # name = 'fx2SAM/plot_' + ppn + '_' + im + '_' + str(fx_no) + '.png'
            # plt.savefig(name)
            # plt.close()

            # crop the fx box
            bbox_coords = [min_x, min_y, max_x, max_y]
            with open('fx2SAM_cropped_BBOXCOORDS/fxboxCOORDS_' + ppn + '_' + im + '_' + str(fx_no) + '.json', 'w') as jf:
                json.dump(bbox_coords, jf)

            # orig_image = Image.fromarray(image, 'RGB')
            # fx_box = orig_image.crop((min_x, min_y, max_x, max_y))
            # fx_box.save('fx2SAM_cropped/fxbox_' + ppn + '_' + im + '_' + str(fx_no) + '.png')

            fx_no += 1

print(count_sample)

