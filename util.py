# any utility functions
# train loop
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn


class RockDataset(Dataset):
    def __init__(self, root_dir, transform, processor):
        self.root_dir = root_dir
        self.transform = transform
        self.processor = processor
        self.classes = sorted(os.listdir(self.root_dir))
        
        self.classes.remove('.DS_Store')
        self.class_to_idx = {cls:idx for idx,cls in enumerate(self.classes)}
        #print(self.class_to_idx)
        self.images_data = self._load_images()
        
    def _load_images(self):
        images = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir,cls)
            for file_name in os.listdir(cls_path):
                image_path = os.path.join(cls_path,file_name)
                images.append((image_path,self.class_to_idx[cls])) 
        return images
        
        
        
    def __getitem__(self,idx):
        #idx = 1
        
        image_path,label = self.images_data[idx]
        
        image = Image.open(image_path)#.convert('RGB') # a sample image is size (474, 386)
        
        if self.transform:
            image = self.transform(image)
        if self.processor:
            image = self.processor(images=image, return_tensors="pt")   
        return image, label        
        
    def __len__(self):
        return len(self.images_data)
    
    
    
