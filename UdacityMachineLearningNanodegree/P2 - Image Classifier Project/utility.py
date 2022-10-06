# For loading data and preprocessing images

# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from PIL import Image

def dataloaders_fun(data_dir):
    
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Use transformations
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ]),
        'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ]),
        'test': transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    batch_size=64
    num_workers=0

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], 
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False),
        'test': torch.utils.data.DataLoader(image_datasets['test'],
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=False)
    }
    
    return image_datasets, dataloaders


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    size = (256, 256)
    pil_image = im.resize(size)
    left = (256-224)/2
    upper = (256-224)/2
    right = (256+224)/2
    lower = (256+224)/2
    pil_image = pil_image.crop((left, upper, right, lower))
    np_image = np.array(pil_image)/256
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_std = (np_image - mean)/std
    np_image_t = torch.from_numpy(np_image_std.transpose((2, 0, 1)))
    
    return np_image_t 


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax






    
