import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from model import LPImageDataset, MultiBoxLoss, training_loop, custom_collate_fn

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.cuda.empty_cache() if device == "cuda" else None

# Load data
path = ".\\processed_images\\data\\"
data = LPImageDataset(path + "annotations.csv", path, device=device)
train_loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, data.num_classes)
    
model.to(device)

# Train model
training_loop(model, train_loader, device=device, epochs=50, lr=0.001)

# # Save model
# torch.save(model.state_dict(), ".\\model.pth")

# # Load model
# model = NeuralNetwork()
# model.load_state_dict(torch.load(".\\model.pth"))

# # Evaluate model
# model.eval()

# # Predict
# image, bbox, label = data[0]
# image = image.unsqueeze(0)
# bbox = bbox.unsqueeze(0)
# label = label.unsqueeze(0)
# print(model(image, bbox, label))