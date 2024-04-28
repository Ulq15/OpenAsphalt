import argparse
<<<<<<< Updated upstream
=======
from sympy import false
>>>>>>> Stashed changes
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.io import read_image
# from preprocess import to_csv, show_bbox
# from pathlib import Path
# import numpy as np
# import os
# import time

def predict(image:str, model:FasterRCNN, device:str):
    model.eval()
<<<<<<< Updated upstream
    image = read_image(image).to(device).float() / 255.0
    pred = model([image])
=======
    img = read_image(image).to(device).float() / 255.0
    pred = model(img)
>>>>>>> Stashed changes
    print(pred)
        

def test(args, model:FasterRCNN, dataset):
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=single_label_collate_fn,
    )
    avg_mAP, avg_p, avg_r = calculate_metrics(model, data_loader)
    print(f"Average MAP: {avg_mAP}")
<<<<<<< Updated upstream
    print(f"Average Precision: {avg_p}")
    print(f'Average Recall: {avg_r}')
=======
    # print(f"Average Precision: {avg_p}")
    # print(f'Average Recall: {avg_r}')
>>>>>>> Stashed changes


def validate(args, model:FasterRCNN, dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=single_label_collate_fn,
    )
    model.eval()
    avg_mAP, avg_p, avg_r = calculate_metrics(model, data_loader)
    print(f"Average MAP: {avg_mAP}")
<<<<<<< Updated upstream
    print(f"Average Precision: {avg_p}")
    print(f'Average Recall: {avg_r}')
=======
    # print(f"Average Precision: {avg_p}")
    # print(f'Average Recall: {avg_r}')
>>>>>>> Stashed changes


def train(args, model:FasterRCNN, dataset):
    # Load data
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=multilabel_collate_fn,
    )
    
    val_loader = DataLoader(
        LPImageDataset(
            args.dataset + "valid\\",
            args.dataset + "valid\\annotations.csv",
            device=args.device,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=multilabel_collate_fn
    )
    
    training_loop(
        model,
        data_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        save_filename=args.save_to
    )
    torch.save(model, '.\\'+args.save_to+'.m')
    torch.save(model.state_dict(), ('.\\'+args.save_to+'.w'))
    
    model.eval()
    avg_mAP = calculate_metrics(model, data_loader)
    print(f"Average MAP on Training Data: {avg_mAP}")


def setup(args):
    # Load device
    if args.device == "cuda" and not (torch.cuda.is_available()):
        print("There is no cuda device available, exiting...")
        return
    elif args.device == "cuda":
        torch.cuda.empty_cache()
    # print(f"Using {args.device} device")

    if args.mode == "pred":
        model = torch.load(args.model_file)
        model.to(device=args.device)
        model.load_state_dict(torch.load(args.weight_file))
<<<<<<< Updated upstream
        return predict(args, model)

=======
        return predict(args.image, model, args.device)
    
>>>>>>> Stashed changes
    # Load data
    dataset = LPImageDataset(
        args.dataset + f"{args.mode}\\",
        args.dataset + f"{args.mode}\\annotations.csv",
        device=args.device,
    )

    # Load model
    # if args.model_file:
    #     model = torch.load(args.model_file)
    # else:
    model = load_model(dataset.num_classes)
    model.to(args.device)

    # Check for weights file
    if args.weight_file:
        model.load_state_dict(torch.load(args.weight_file))

    if args.mode == "train":
        train(args, model, dataset)
    elif args.mode == "valid":
        validate(args, model, dataset)
    elif args.mode == "test":
        test(args, model, dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("--mode", default="val", help="one of ['train', 'valid', 'test', 'pred']")
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
        type=int,
        default=10,
        help="the number steps for the optimizer to apply learning rate decay",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        help="save model & weights to 2 files during training, with .m for model file & .w for weights",
    )
    parser.add_argument("--image", type=str, default="", help="Path to single image for the model to evaluate")
    # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()

    setup(opt)
