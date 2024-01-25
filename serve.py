import torch
import torch.nn as nn
import timm # pytorch library for image models
from transformers import AutoFeatureExtractor, ViTFeatureExtractor,ViTImageProcessor, ViTForImageClassification
from torchvision import transforms
from util import RockDataset
from PIL import Image
from torch.utils.data import DataLoader



processor = ViTImageProcessor.from_pretrained('/Users/nadiapasha/ML_prep/projects/RockClassification/rock_model')
model = ViTForImageClassification.from_pretrained('/Users/nadiapasha/ML_prep/projects/RockClassification/rock_model')

image = Image.open('/Users/nadiapasha/ML_prep/projects/RockClassification/test/Amphibolite/Amphibolite4.png')
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])