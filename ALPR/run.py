import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import *
from preprocess import to_csv, show_bbox

def test(args):
    pass

def validate(args):
    pass

def train(args):
    pass

def setup(args):
    # Load device
    if args.device == 'cuda' and not (torch.cuda.is_available()):
        print("There is no cuda device available, exiting...")
        return
    elif args.device == "cuda":
       torch.cuda.empty_cache()
    print(f"Using {args.device} device")
    
    # Check for weights file
    if args.weight_file is None:
        if args.resume:
            print("Cannot continue training without a weights .pth file specified")
            return
        args.weight_file = os.path.join('.\\weights.pth')
        print('Since weights file was not given, setting weights to .\\weights.pth')
    time.localtime(time.time())     #<-------- name of new weights file
    
    # Load data
    dataset = LPImageDataset(args.dataset + "annotations.csv", args.dataset, device=args.device)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

    # Load model
    model = load_model(dataset.num_classes)
    model.load_state_dict(torch.load(args.weight_file))
    model.to(args.device)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'val':
        validate(args)
    elif args.mode == 'test':
        test(args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='run.py')
    # parser.add_argument('--mode', default='val', help="one of ['train', 'val', 'test']")
    # parser.add_argument('--dataset', type=str, default='.\\processed_images\\data\\', help='dataset directory')
    parser.add_argument('--weight_file', type=str, help='load weights from a .pth file. if using --resume, weights are overwritten, otherwise weights will be saved to a file with named by Date & Time')    
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Continue training using loaded weights file and overwrites it')
    # parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    # parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    # parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()
    
    setup(opt)


