import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import json
import argparse
from utility import *
from model_functions import *

def main():

    parser = argparse.ArgumentParser(description='Train a new network')

    parser.add_argument('data_dir', type=str,
                        help='Path of the Image Dataset with train, valid and test subfolders)')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='Directory for saving model checkpoints')
    parser.add_argument('--arch', type=str, default='resnet101',
                        help='Models architeture from torchvision.models')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of epochs')
    parser.add_argument('--hidden1', type=int, default=2048,
                        help='Hidden units for first layer')
    parser.add_argument('--hidden2', type=int, default=512,
                        help='Hidden units for second layer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gpu', type=str, default='gpu',
                        help='Use gpu or cpu')
    
    args = parser.parse_args()

    data_dir = args.data_dir

    save_dir = args.save_dir

    arch = args.arch

    epochs = args.epochs
        
    hidden1 = args.hidden1
        
    hidden2 = args.hidden2

    learning_rate = args.learning_rate

    if args.gpu=='gpu' and torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU")
    elif args.gpu=='gpu' and not torch.cuda.is_available():
        device = 'cpu'
        print("GPU not available. Using CPU")
    else:
        device='cpu'
        print("Using CPU")
    
    image_datasets, dataloaders = dataloaders_fun(data_dir)
    
    traindata=image_datasets['train']
    validdata=image_datasets['valid']
    testdata=image_datasets['test']
    
    trainloader=dataloaders['train']
    validloader=dataloaders['valid']
    testloader=dataloaders['test']

    print(args)
    print("\n\n")
    
    model=train(trainloader, validloader, architecture=arch, epochs=epochs,
          hidden1=hidden1, hidden2=hidden2, learn_rate=learning_rate, device=device) 

    save_checkpoint(model=model, architecture=arch, modelname=arch, save_dir=save_dir, filename='checkpt.pth', image_train=traindata)
    
if __name__ == '__main__':
    main()