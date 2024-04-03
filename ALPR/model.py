import os
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import  transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class LPImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device='cpu'):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.label_mapping = {val: i for i, val in enumerate(self.annotations.iloc[:,5].unique())}
        self.annotations.iloc[:,5] = self.annotations.iloc[:,5].map(self.label_mapping)
        self.num_classes = len(self.label_mapping)
        self.device = device
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0]) 
        image = read_image(img_path).to(self.device)
        image = image.float() / 255.0
        annotation = {'boxes': torch.tensor(self.annotations.iloc[idx, 1:5].values.astype(int)).to(self.device), 
                      'labels': torch.tensor([self.annotations.iloc[idx, 5]]).to(self.device)}
        # bbox = torch.tensor(self.annotations.iloc[idx, 1:5].values.astype(int).tolist()).to(self.device)
        # label = torch.tensor(self.annotations.iloc[idx, 5]).to(self.device)
        # print(image.shape, annotation)
        return image, annotation

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh=0.5):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.loc_loss = nn.SmoothL1Loss()
        self.conf_loss = nn.CrossEntropyLoss()
    
    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        """
        loc_preds: predicted bounding box offsets (batch_size, num_priors * 4)
        loc_targets: ground truth bounding box offsets (batch_size, num_priors * 4)
        conf_preds: predicted class scores (batch_size, num_priors, num_classes)
        conf_targets: ground truth class labels (batch_size, num_priors)
        """
        pos = conf_targets > 0  # Positive priors
        num_pos = pos.long().sum(1, keepdim=True)
        
        # Localization Loss
        loc_loss = self.loc_loss(loc_preds[pos], loc_targets[pos])
        
        # Confidence Loss (with hard negative mining)
        conf_loss = F.cross_entropy(conf_preds.view(-1, self.num_classes), conf_targets.view(-1), ignore_index=-1)
        
        return loc_loss, conf_loss

def custom_collate_fn(batch):
    boxes =[sample['boxes'] for _,sample in batch]
    labels = [sample['labels'] for _,sample in batch]
    targets =  [{'boxes': box.clone().detach().reshape(1,len(box)), 'labels': label.clone().detach()} for box, label in zip(boxes, labels)]
    tensors = torch.stack([sample for sample, _ in batch])
    return tensors, targets
            
def training_loop(model:nn.Module, train_loader:DataLoader, device='cpu', epochs=50, lr=0.001):
    # Initialize model and loss
    # num_classes = train_loader.dataset.num_classes
    # criterion = MultiBoxLoss(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        total_loc_loss = 0.0
        total_conf_loss = 0.0
        running_loss = 0.0
                
        print(f"=== Strating Epoch {epoch+1}/{num_epochs} ===")
        for images, targets in train_loader:
            images = Variable(images).to(device)
            # loc_targets = [target['boxes'] for target in targets]
            # conf_targets = [target['labels'] for target in targets]
            optimizer.zero_grad()   
            
            loss_dict  = model(images, targets)
            # loc_loss, conf_loss = criterion(loss_dict['boxes'], loc_targets, loss_dict['labels'], conf_targets)
            # total_loc_loss += loc_loss.item()
            # total_conf_loss += conf_loss.item()
            # loss = loc_loss + conf_loss
            # loss.backward()
            # optimizer.step()
            
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        # Print training statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_loss}')