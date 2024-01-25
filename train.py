# adding a linear head to the model to train
from concurrent.futures import process
import torch
import torch.nn as nn
import time
from transformers import ViTImageProcessor, ViTForImageClassification, AutoConfig
from util import RockDataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging


logging.basicConfig(
    filemode="w",
    filename="./errors-{}.log".format(int(time.time())),
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

overall_best = float("inf")

parameterization = {'lr':0.01, 'batch_size':4}
num_classes = 53

base_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
dataset = RockDataset(root_dir='/Users/nadiapasha/ML_prep/projects/RockClassification/Rocks', transform = None, processor = processor)#transform=transform)


# Load the original configuration
original_config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
# Modify the configuration as needed
modified_config = original_config
modified_config.label2id = dataset.class_to_idx 
modified_config.id2label = {v:k for k,v in dataset.class_to_idx.items()}
new_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=modified_config, ignore_mismatched_sizes=True)

for param in new_model.parameters():
    param.requires_grad = False   
    
new_model.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

for param in new_model.classifier.parameters():
    param.requires_grad = True



# Loss function:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(new_model.parameters(),lr = 0.001)

batch_size = 4
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


num_epochs = 10
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('device',device)
new_model = new_model.to(device)


def main():
    
    params = "RockModel-" + ",".join(
        ["{}={}".format(k, v) for k, v in parameterization.items()]
    )
    writer = SummaryWriter("logs/{}".format(params))
    logging.info(params)
    
    global_step = 0
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images['pixel_values'] = images['pixel_values'].squeeze()
            images = images.to(device)
            labels = labels.to(device)
            outputs = new_model(**images)
            logits = outputs.logits
            loss = criterion(logits, labels)
            #pred = logits.argmax(-1)
            
        #     # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('Loss', loss.item(), global_step)
        #writer.add_scalar('Accuracy', accuracy, global_step)

        # Increment the global step
        global_step += 1
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    new_model.save_pretrained("rock_model")
    processor.save_pretrained('rock_model')
    

if __name__ == '__main__':
    main()