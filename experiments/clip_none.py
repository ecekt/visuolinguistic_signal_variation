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
_, preprocess = clip.load("ViT-B/32", device=device)

# torch.all(preprocess(Image.open(image_path)) ==
# processor.image_processor(Image.open(image_path), return_tensors='pt').pixel_values.squeeze())
# True

from transformers import CLIPConfig, CLIPModel

# Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
configuration = CLIPConfig()

# Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
model = CLIPModel(configuration)

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
                image_features = model.get_image_features(image)

                clip_preprocessed_images[img_id] = image_features.to(torch.device("cpu"))

print(count_img, len(clip_preprocessed_images))

with open('data/didec/clip_NONE_preprocessed_didec_images_cpu.pickle', 'wb') as f:
    pickle.dump(clip_preprocessed_images, f)

