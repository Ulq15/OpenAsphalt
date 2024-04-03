import os
# import numpy as np
import cv2
import pandas as pd
# from sklearn.preprocessing import minmax_scale
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
# from torchvision import datasets, transforms
# from torchvision.datasets import ImageFolder

class LPImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device='cpu'):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.label_mapping = {val: i for i, val in enumerate(self.annotations.iloc[:,5].unique())}
        self.annotations.iloc[:,5] = self.annotations.iloc[:,5].map(self.label_mapping)
        self.device = device
        # print(self.annotations.head())
        # print(len(self.label_mapping))
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0]) # type: ignore
        image = read_image(img_path).to(self.device)
        image = image.float() / 255.0
        bbox = torch.tensor(self.annotations.iloc[idx, 1:5].values.astype(int)).to(self.device)
        label = torch.tensor(self.annotations.iloc[idx, 5]).to(self.device)
        return image, bbox, label

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

class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        
        # Backbone (VGG for example)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 768, padding=1),  # Example: VGG block 1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            # Add more layers as needed
        )
        
        # Extra layers for detection
        self.extra = nn.Sequential(
            nn.Conv2d(1024, 256, 1),  # Example: Additional conv layer
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            # Add more layers as needed
        )
        
        # Detection head
        self.loc = nn.Sequential(
            nn.Conv2d(512, 4 * num_classes, 3, padding=1),  # 4 bounding box coordinates per class
        )
        self.conf = nn.Sequential(
            nn.Conv2d(512, num_classes, 3, padding=1),  # Confidence score for each class
        )
    
    def forward(self, x):
        sources = []
        loc = []
        conf = []
        # Backbone
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i == 3 or i == 5:  # Save some intermediate layers as sources
                sources.append(x)
        # Extra layers
        for layer in self.extra:
            x = layer(x)
            sources.append(x)
        # Apply detection head to each source
        for source in sources:
            loc.append(self.loc(source).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4))
            conf.append(self.conf(source).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes))
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return loc, conf

class ResNetSSD(nn.Module):
    def __init__(self, num_classes):
        super(ResNetSSD, self).__init__()
        self.num_classes = num_classes
        
        # Load a pre-trained ResNet model
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # print(resnet)
        # Remove fully connected layers (FC)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Extra layers for detection
        self.extra = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Detection head
        self.loc = nn.Sequential(
            nn.Conv2d(64, 4 * num_classes, 3, padding=1),  # 4 bounding box coordinates per class
        )
        self.conf = nn.Sequential(
            nn.Conv2d(64, num_classes, 3, padding=1),  # Confidence score for each class
        )
    
    def forward(self, x):
        sources = []
        loc = []
        conf = []
        
        # Backbone
        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                sources.append(x)
        
        # Extra layers
        for layer in self.extra:
            x = layer(x)
            sources.append(x)
        
        # Apply detection head to each source
        for source in sources:
            print("Loc:  ", self.loc)
            print("Size: ", source.size())
            loc.append(self.loc(source).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4))
            conf.append(self.conf(source).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes))
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        return loc, conf

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

def training_loop(model, train_loader, device='cpu', epochs=50, lr=0.001):
    # Initialize model and loss
    if model is None:
        model = ResNetSSD(num_classes=213).to(device)  # 2 classes: background and license plate
    criterion = MultiBoxLoss(num_classes=213).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        total_loc_loss = 0.0
        total_conf_loss = 0.0
        
        print(f"=== Strating Epoch {epoch+1}/{num_epochs} ===")
        for images, targets in train_loader:
            images = Variable(images)
            # loc_targets = Variable(loc_targets)
            # conf_targets = Variable(conf_targets)
            loc_targets = [target['boxes'] for target in targets]
            conf_targets = [target['labels'] for target in targets]
            optimizer.zero_grad()
            loc_preds, conf_preds = model(images)
            
            loc_loss, conf_loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
            total_loc_loss += loc_loss.item()
            total_conf_loss += conf_loss.item()
            
            loss = loc_loss + conf_loss
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Print training statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Loc Loss: {total_loc_loss}, Conf Loss: {total_conf_loss}')

def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    images = []
    annotations = []
    for index, row in df.iterrows():
        image = cv2.imread(row['filename'])
        images.append(image)
        # Bounding box coordinates (x_min, y_min, x_max, y_max)
        bbox = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
        class_label = row['class']
        annotations.append({'bbox': bbox, 'class': class_label})
    return images, annotations

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    if device == "cuda":
        torch.cuda.empty_cache()
    path = ".\\processed_images\\data\\"
    model = SSD(num_classes=213).to(device)
    # model2 = ResNetSSD(num_classes=213).to(device)
    # data = load_dataset(path + "annotations.csv")  # Load your dataset here
    data = LPImageDataset(path + "annotations.csv", path, device=device)
    train_loader = DataLoader(data, batch_size=16, shuffle=True)
    # trainDataset = ImageFolder(path)
    # valDataset = ImageFolder(path)
    # trainDataLoader  = DataLoader(data, batch_size=16, shuffle=True)
    # valDataLoader = DataLoader(valDataset, batch_size=16)
    training_loop(model, train_loader, device, epochs=2)
    
    # torch.cuda.OutOfMemoryError: CUDA out of memory.
    # Tried to allocate 1.12 GiB. GPU 0 has a total capacity of 12.00 GiB of which 0 bytes is free. 
    # Of the allocated memory 33.03 GiB is allocated by PyTorch, and 891.54 MiB is reserved by PyTorch but unallocated. 
    # If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 
    # See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    
    # model = NeuralNetwork().to(device)
    # print(model)
    
    # X = torch.rand(1, 28, 28, device=device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    
    # ##########################################################
    # # Example of what happens inside a model 
    # ##########################################################
    
    # input_image = torch.rand(3,28,28)
    # print(input_image.size())
    # print("input_image: ", input_image, '\n\n')
    
    # ##### this happens inside the model from here #####
    # flatten = nn.Flatten()
    # flat_image = flatten(input_image)
    # print(flat_image.size())
    # print("Flat Image: ", flat_image, '\n\n')
    
    # layer1 = nn.Linear(in_features=28*28, out_features=20)
    # hidden1 = layer1(flat_image)
    # print(hidden1.size())
    
    # print(f"Before ReLU: {hidden1}\n\n")
    # hidden1 = nn.ReLU()(hidden1)
    # print(f"After ReLU: {hidden1}")


    # seq_modules = nn.Sequential(
    #     flatten,
    #     layer1,
    #     nn.ReLU(),
    #     nn.Linear(20, 10)
    # )
    # input_image = torch.rand(3,28,28)
    # logits = seq_modules(input_image)
    
    # softmax = nn.Softmax(dim=1)
    # pred_probab = softmax(logits)
    # ##### to here #####
    
    # ##########################################################
    # # Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training.
    # # Subclassing nn.Module automatically tracks all fields defined inside your model object, 
    # # makes all parameters accessible using your model’s parameters() or named_parameters() methods.
    # ##########################################################
    # print(f"Model structure: {model}\n\n")

    # for name, param in model.named_parameters():
    #     print(f"[\nLayer:\t{name}\nSize:\t{param.size()}\nValues:\t{param[:2]}\n]\n")

    # ResNet(
    #   (conv1): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   (relu): ReLU(inplace=True)
    #   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #   (layer1): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #   )
    #   (layer2): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (3): Bottleneck(
    #       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #   )
    #   (layer3): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (3): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (4): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (5): Bottleneck(
    #       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #   )
    #   (layer4): Sequential(
    #     (0): Bottleneck(
    #       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (downsample): Sequential(
    #         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): Bottleneck(
    #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #     (2): Bottleneck(
    #       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #     )
    #   )
    #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    #   (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # )