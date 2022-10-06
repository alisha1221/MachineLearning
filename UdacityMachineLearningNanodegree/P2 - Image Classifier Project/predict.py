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

    parser = argparse.ArgumentParser(description='Prediction')

    parser.add_argument('inputpath', type=str,
                        help='Path of the Image')
    parser.add_argument('checkpoint', type=str, default='./checkpoint.pth',
                        help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, 
                        help='Top k predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Category names')
    parser.add_argument('--gpu', type=str, default='gpu',
                        help='Use gpu or cpu')
    
    args = parser.parse_args()
    
    inputpath = args.inputpath
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names

    if args.gpu=='gpu' and torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU")
    elif args.gpu=='gpu' and not torch.cuda.is_available():
        device = 'cpu'
        print("GPU not available. Using CPU")
    else:
        device='cpu'
        print("Using CPU")
        
    print(args)
    
    # Fixed cuda runtime error from here: https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    
    if device=='cuda':
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    model_checkpoint = torch.load(checkpoint, map_location=map_location)
  
    model = load_checkpoint(filepath=checkpoint, architecture=model_checkpoint['architecture'], map_location=map_location)

    result=predict(image_path=inputpath, model=model, topk=top_k, category_names=category_names, device=device)
    
    print(result)
    
if __name__ == '__main__':
    main()