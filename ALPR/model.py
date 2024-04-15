import os
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.io import read_image
from torchvision import  transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class LPImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device='cpu'):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            # transforms.Lambda(lambd=resize_with_pad),
            # transforms.Resize(76,max_size=768),
            # transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float)
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
        image = self.transform(image)
        image = image.float() / 255.0
        annotation = {'boxes': torch.tensor(self.annotations.iloc[idx, 1:5].values.astype(dtype=int)).to(self.device), 
                      'labels': torch.tensor([self.annotations.iloc[idx, 5]]).to(self.device)}
        # bbox = torch.tensor(self.annotations.iloc[idx, 1:5].values.astype(int).tolist()).to(self.device)
        # label = torch.tensor(self.annotations.iloc[idx, 5]).to(self.device)
        # print(image.shape, annotation)
        return image, annotation


def custom_collate_fn(batch):
    boxes =[sample['boxes'] for _,sample in batch]
    labels = [sample['labels'] for _,sample in batch]
    targets =  [{'boxes': box.clone().detach().reshape(1,len(box)), 'labels': label.clone().detach()} for box, label in zip(boxes, labels)]
    tensors = torch.stack([sample for sample, _ in batch])
    return tensors, targets
            
def training_loop(model:nn.Module, train_loader:DataLoader, device='cpu', epochs=50, lr=0.001): 
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        # total_loc_loss = 0.0
        # total_conf_loss = 0.0
        running_loss = 0.0
                
        print(f"=== Strating Epoch {epoch+1}/{num_epochs} ===")
        for images, targets in train_loader:
            images = Variable(images).to(device)
            # loc_targets = [target['boxes'] for target in targets]
            # conf_targets = [target['labels'] for target in targets]
            optimizer.zero_grad()   
            
            ''' The model returns a Dict[Tensor] during training, containing the classification and regression 
                losses for both the RPN and the R-CNN.
                Format:
                dict = {
                    'loss_classifier': tensor(5.3523, device='cuda:0', grad_fn=<NllLossBackward0>), 
                    'loss_box_reg': tensor(0.0001, device='cuda:0', grad_fn=<DivBackward0>), 
                    'loss_objectness': tensor(0.6812, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
                    'loss_rpn_box_reg': tensor(0.0018, device='cuda:0', grad_fn=<DivBackward0>)
                }
            '''
            loss_dict  = model(images, targets)
            # print(str(loss_dict))
            
            # loc_loss, conf_loss = criterion(loss_dict['boxes'], loc_targets, loss_dict['labels'], conf_targets)
            # total_loc_loss += loc_loss.item()
            # total_conf_loss += conf_loss.item()
            # loss = loc_loss + conf_loss
            # loss.backward()
            # optimizer.step()
            
            losses = torch.stack([loss for loss in loss_dict.values()])
            losses = torch.sum(losses)
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        # Print training statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_loss}')

