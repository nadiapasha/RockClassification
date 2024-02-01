import torch
import torch.nn as nn
#import timm # pytorch library for image models
from transformers import AutoFeatureExtractor, ViTFeatureExtractor,ViTImageProcessor, ViTForImageClassification
from torchvision import transforms
from util import RockDataset
from PIL import Image
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix
import re



processor = ViTImageProcessor.from_pretrained('/Users/nadiapasha/ML_prep/projects/Rock/rock_model_small')
model = ViTForImageClassification.from_pretrained('/Users/nadiapasha/ML_prep/projects/Rock/rock_model_small')

y_true = []
y_pred = []
labels = []
for class_ in os.listdir('/Users/nadiapasha/ML_prep/projects/Rock/test_data'):
    if class_ != '.DS_Store':
        labels.append(class_)
        parent_dir = '/Users/nadiapasha/ML_prep/projects/Rock/test_data/'+class_
        
        for item in os.listdir(parent_dir): 
            if item != '.DS_Store':

                image = Image.open(parent_dir+'/'+item)
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                # model predicts one of the classes
                predicted_class_idx = logits.argmax(-1).item()
                true = re.sub(r"\d+\.png", '', item)
                y_true.append(true)
                y_pred.append(model.config.id2label[predicted_class_idx])
            #print("Predicted class:", model.config.id2label[predicted_class_idx])
print('Labels: ',labels)
print(confusion_matrix(y_true, y_pred, labels=labels))