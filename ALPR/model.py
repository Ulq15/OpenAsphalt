import time
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from imutils import paths

# import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# import torchvision
from torchvision.io import read_image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(progress=True, num_classes=num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class LPImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, device="cpu"):
        self.img_dir = img_dir
        if annotations_file is not None:
            self.annotations = pd.read_csv(annotations_file)
            self.label_mapping = {
                val: i for i, val in enumerate(self.annotations.iloc[:, 5].unique())
            }
            self.key_by_label = {i: val for val, i in self.label_mapping.items()}
            self.annotations.iloc[:, 5] = self.annotations.iloc[:, 5].map(
                self.label_mapping
            )
            self.num_classes = len(self.label_mapping)
        # self.transform = transforms.Compose([
        #     transforms.Grayscale(3),
        # #     transforms.Lambda(lambd=resize_with_pad),
        #     transforms.Resize(76,max_size=768),
        #     transforms.ToTensor(),
        #     transforms.ConvertImageDtype(torch.float)
        # ])
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path).to(self.device)
        # image = self.transform(image)
        image = image.float() / 255.0
        annotation = {
            "boxes": torch.tensor(
                self.annotations.iloc[idx, 1:5].values.astype(dtype=int)
            ).to(self.device),
            "labels": torch.tensor([self.annotations.iloc[idx, 5]]).to(self.device),
        }
        # bbox = torch.tensor(self.annotations.iloc[idx, 1:5].values.astype(int).tolist()).to(self.device)
        # label = torch.tensor(self.annotations.iloc[idx, 5]).to(self.device)
        # print(image.shape, annotation)
        return image, annotation


class UnlabeledLPImageDataset(Dataset):
    def __init__(self, img_dir, device="cpu"):
        self.img_dir = img_dir
        self.images = sorted(list(paths.list_images(img_dir)))
        self.device = device
        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            # transforms.Lambda(lambd=resize_with_pad),
            # transforms.Resize(76,max_size=768),
            # transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float)
        ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path).to(self.device)
        image = self.transform(image)
        image = image.float() / 255.0
        return image


def custom_collate_fn(batch):
    boxes = [sample["boxes"] for _, sample in batch]
    labels = [sample["labels"] for _, sample in batch]
    targets = [
        {
            "boxes": box.clone().detach().reshape(1, len(box)),
            "labels": label.clone().detach(),
        }
        for box, label in zip(boxes, labels)
    ]
    tensors = torch.stack([sample for sample, _ in batch])
    return tensors, targets


def training_loop(model: nn.Module, train_loader: DataLoader, device="cpu", epochs=50, lr=0.001, step_size=10, model_file='.\\model', weight_file='.\\weights'):
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Training loop
    model.train()
    num_epochs = epochs
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0

        print(f"=== Strating Epoch {epoch+1}/{num_epochs} ===")
        for images, targets in train_loader:
            images = Variable(images).to(device)
            optimizer.zero_grad()

            loss_dict = model(images, targets)

            losses = torch.stack([loss for loss in loss_dict.values()])
            losses = torch.sum(losses)
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        # Print training statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_loss}")
        print(f"({time.strftime('%X')}):: Epoch took about {int((time.time() - start)/60)} minutes to complete.")

        if (epoch + 1) % step_size == 0:
            print(f"Saving model...")
            torch.save(model, model_file)
            torch.save(model.state_dict(), weight_file if weight_file else ".\\weights")
            print(f"Saving Done.")


def calculate_mAP(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    """
    Calculate Mean Average Precision (mAP) for object detection.

    Parameters:
        gt_boxes (numpy array): Ground truth bounding boxes of shape (N, 4).
        gt_labels (numpy array): Ground truth class labels of shape (N,).
        pred_boxes (numpy array): Predicted bounding boxes of shape (M, 4).
        pred_scores (numpy array): Predicted scores/confidences of shape (M,).
        pred_labels (numpy array): Predicted class labels of shape (M,).
        iou_threshold (float): IoU threshold for matching predictions to ground truth.

    Returns:
        mAP (float): Mean Average Precision.
    """
    # Sort predictions by confidence scores (descending order)
    sort_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sort_indices]
    pred_scores = pred_scores[sort_indices]
    pred_labels = pred_labels[sort_indices]

    # Initialize variables
    true_positives = np.zeros(len(pred_boxes))
    false_positives = np.zeros(len(pred_boxes))
    num_gt_boxes = len(gt_boxes)
    matched_indices = set()

    # Loop over predictions
    for i, pred_box in enumerate(pred_boxes):
        ious = compute_iou(pred_box, gt_boxes)
        max_iou = np.max(ious)
        max_iou_idx = np.argmax(ious)

        if (
            max_iou >= iou_threshold
            and max_iou_idx not in matched_indices
            and gt_labels[max_iou_idx] == pred_labels[i]
        ):
            true_positives[i] = 1
            matched_indices.add(max_iou_idx)
        else:
            false_positives[i] = 1

    # Compute precision and recall
    cumsum_tp = np.cumsum(true_positives)
    cumsum_fp = np.cumsum(false_positives)
    precision = cumsum_tp / (cumsum_tp + cumsum_fp)
    recall = cumsum_tp / num_gt_boxes

    # Compute AP using precision-recall curve (area under curve)
    ap = compute_ap(precision, recall)
    return ap


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes.

    Parameters:
        box1, box2 (numpy arrays): Bounding boxes in format [x1, y1, x2, y2].

    Returns:
        iou (float): IoU value.
    """
    x1 = max(box1[0], box2[:, 0])
    y1 = max(box1[1], box2[:, 1])
    x2 = min(box1[2], box2[:, 2])
    y2 = min(box1[3], box2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area_box1 + area_box2 - intersection

    iou = intersection / union
    return iou


def compute_ap(precision, recall):
    """
    Compute Average Precision (AP) from precision-recall curve.

    Parameters:
        precision, recall (numpy arrays): Precision and recall values.

    Returns:
        ap (float): Average Precision.
    """
    # Append end points to the precision and recall arrays
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    # Compute area under curve
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap


def calculate_metrics(model:nn.Module, test_loader):
    # model.eval()
    mAPs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image for image in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Calculate mAP for each image
                gt_boxes = targets[i]["boxes"].cpu().numpy()
                gt_labels = targets[i]["labels"].cpu().numpy()

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                mAP = calculate_mAP(
                    gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels
                )
                mAPs.append(mAP)

    avg_mAP = np.mean(mAPs)
    return avg_mAP
