from PIL import Image
import torch
import clip
from transformers import CLIPProcessor, CLIPModel

image_path = "000000039769.jpg"

modelhf = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = processor.image_processor(Image.open(image_path), return_tensors='pt').pixel_values

with torch.no_grad():
    image_features = modelhf.get_image_features(image)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image2 = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

with torch.no_grad():
    image_features2 = model.encode_image(image2)

print()

assert torch.all(image == image2)
# assert torch.all(image_features == image_features2)
# very close but not exactly the same diff  e-08
