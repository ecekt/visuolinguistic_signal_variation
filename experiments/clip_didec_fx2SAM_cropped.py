import os
import torch
import clip
from PIL import Image
from collections import defaultdict
import pickle

clip_preprocessed_fxs = defaultdict(dict)

# dtype on cuda float16, on cpu float32
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model_type = "ViT-L/14"
model_type = "ViT-B/32"
print(model_type)
model, preprocess = clip.load(model_type, device=device)

count_img = 0

for root, dirs, files in os.walk('/home/ece/Desktop/phd_2022/gaze_code/segment/fx2SAM_cropped'):

    for f in files:

        if '.png' in f:
            count_img += 1
            if count_img % 10 == 0:
                print(count_img)

            _, ppn, img_id, fx_id = f.split('.')[0].split('_')

            image_path = os.path.join(root, f)

            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

                clip_preprocessed_fxs[(ppn, img_id)][fx_id] = image_features.to(torch.device("cpu"))

print(count_img, len(clip_preprocessed_fxs))

if model_type == "ViT-L/14":
    print('saving clip_preprocessed_didec_fx2SAMcropped_L14_cpu')
    with open('data/didec/clip_preprocessed_didec_fx2SAMcropped_L14_cpu.pickle', 'wb') as f:
        pickle.dump(clip_preprocessed_fxs, f)
elif model_type == "ViT-B/32":
        print('saving clip_preprocessed_didec_fx2SAMcropped_B32_cpu')
        with open('data/didec/clip_preprocessed_didec_fx2SAMcropped_B32_cpu.pickle', 'wb') as f:
            pickle.dump(clip_preprocessed_fxs, f)
