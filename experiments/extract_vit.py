from transformers import ViTImageProcessor, ViTModel
import torch
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
import os
import pickle

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

vit_preprocessed_images = defaultdict()

count_img = 0

for root, dirs, files in os.walk('data/didec/images'):

    for f in files:

        if '.jpg' in f:
            count_img += 1
            if count_img % 10 == 0:
                print(count_img)

            img_id = f.split('.')[0]

            image_path = os.path.join(root, f)

            image = Image.open(image_path)

            inputs = image_processor(image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

                last_hidden_states = outputs.last_hidden_state
                list(last_hidden_states.shape)

                # [CLS] token
                image_features = last_hidden_states[0][0].unsqueeze(0)

                vit_preprocessed_images[img_id] = image_features.to(torch.device("cpu"))

print(count_img, len(vit_preprocessed_images))

with open('data/didec/vit_preprocessed_didec_images_cpu.pickle', 'wb') as f:
    pickle.dump(vit_preprocessed_images, f)

