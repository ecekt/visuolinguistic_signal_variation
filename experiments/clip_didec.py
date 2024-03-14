import os
import torch
import clip
from PIL import Image
from collections import defaultdict
import pickle


clip_preprocessed_images = defaultdict()

# dtype on cuda float16, on cpu float32
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

count_img = 0

for root, dirs, files in os.walk('data/didec/images'):

    for f in files:

        if '.jpg' in f:
            count_img += 1
            if count_img % 10 == 0:
                print(count_img)

            img_id = f.split('.')[0]

            image_path = os.path.join(root, f)

            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

                clip_preprocessed_images[img_id] = image_features.to(torch.device("cpu"))

print(count_img, len(clip_preprocessed_images))

with open('data/didec/clip_preprocessed_didec_images_cpu.pickle', 'wb') as f:
    pickle.dump(clip_preprocessed_images, f)

