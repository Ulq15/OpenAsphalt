import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from model import *
from preprocess import to_csv, show_bbox
from pathlib import Path
import numpy as np
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.io import read_image

def predict(args, model:nn.Module):
    model.eval()
    args.image
    image = read_image(args.image).to(args.device).float() / 255.0
    pred = model([image])
    print(pred)
    

def test(args, model:nn.Module, dataset):
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    avg_mAP = calculate_metrics(model, data_loader)
    print(f"Average MAP: {avg_mAP}")


def validate(args, model:nn.Module, dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    model.eval()
    avg_mAP = calculate_metrics(model, data_loader)
    print(f"Average MAP: {avg_mAP}")


def train(args, model:nn.Module, dataset):
    # Load data
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    model.train()
    training_loop(
        model,
        data_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        model_file=args.model_file,
        weight_file=args.weight_file,
    )


def setup(args):
    # Load device
    if args.device == "cuda" and not (torch.cuda.is_available()):
        print("There is no cuda device available, exiting...")
        return
    elif args.device == "cuda":
        torch.cuda.empty_cache()
    # print(f"Using {args.device} device")

    if args.mode == "pred":
        model = load_model(2).to(device=args.device)
        model.load_state_dict(torch.load(args.weight_file))
        return predict(args, model)


    # Load data
    dataset = LPImageDataset(
        args.dataset + f"{args.mode}\\",
        args.dataset + f"{args.mode}\\annotations.csv",
        device=args.device,
    )

    # Load model
    if args.model_file:
        model = torch.load(args.model_file)
    else:
        model:FasterRCNN = load_model(dataset.num_classes)
    model.to(args.device)

    # Check for weights file
    if args.weight_file:
        model.load_state_dict(torch.load(args.weight_file))

    if args.mode == "train":
        train(args, model, dataset)
    elif args.mode == "val":
        validate(args, model, dataset)
    elif args.mode == "test":
        test(args, model, dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("--mode", default="val", help="one of ['train', 'val', 'test', 'pred']")
    parser.add_argument(
        "--dataset",
        type=str,
        default=".\\processed_images\\License_Plate_tensorflow\\",
        help="dataset directory",
    )
    parser.add_argument(
        "--weight_file",
        type=str,
        help="load weights from a file. If none, then weights will be saved to a file with named by Date & Time",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="load a model from a file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="size of each image batch"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument(
        "--step_size",
        type=float,
        default=10,
        help="the number steps for the optimizer to apply learning rate decay",
    )
    parser.add_argument("--image", type=str, default="", help="Path to single image for the model to evaluate")
    # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()

    setup(opt)
