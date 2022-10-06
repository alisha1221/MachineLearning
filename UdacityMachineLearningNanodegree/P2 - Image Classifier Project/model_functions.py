# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from torch import optim
from collections import OrderedDict
import json
from utility import *

def train(trainloader, validloader, architecture, epochs, hidden1, hidden2, learn_rate, device):
       
    t_models = {'vgg16': models.vgg16(pretrained=True),
                'densenet121': models.densenet121(pretrained=True),
                'resnet101': models.resnet101(pretrained=True)}
    
    print("Device:", device)
    
    # If unknown architecture is supplied, the model uses resnet101
    if architecture in list(t_models.keys()):
        print('Architecture used: ', architecture)
        model= t_models.get(architecture)
    else:
        print("Model not found. Please try 'vgg16', 'densenet121' or 'resnet101'. \n\nTrying resnet101.")
        model= t_models.get('resnet101')

    #  Freeze parameters
    for param in model.parameters():
        param.requires_grad=False

#     # update classifier
    if architecture=='vgg16':
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(25088, hidden1, bias=True)),
                                 ('relu1', nn.ReLU()),
                                 ('dropout1', nn.Dropout(0.2)),
                                 ('fc2', nn.Linear(hidden1, hidden2, bias=True)),
                                 ('relu2', nn.ReLU()),
                                 ('dropout2', nn.Dropout(0.2)),
                                 ('fc3', nn.Linear(hidden2, 102, bias=True)),
                                 ('output', nn.LogSoftmax(dim=1))
                                 ]))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        
    elif architecture=='densenet121':
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(1024, hidden1, bias=True)),
                                 ('relu1', nn.ReLU()),
                                 ('dropout1', nn.Dropout(0.2)),
                                 ('fc2', nn.Linear(hidden1, hidden2, bias=True)),
                                 ('relu2', nn.ReLU()),
                                 ('dropout2', nn.Dropout(0.2)),
                                 ('fc3', nn.Linear(hidden2, 102, bias=True)),
                                 ('output', nn.LogSoftmax(dim=1))
                                 ]))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
 
    else:
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(2048, hidden1, bias=True)),
                                 ('relu1', nn.ReLU()),
                                 ('dropout1', nn.Dropout(0.2)),
                                 ('fc2', nn.Linear(hidden1, hidden2, bias=True)),
                                 ('relu2', nn.ReLU()),
                                 ('dropout2', nn.Dropout(0.2)),
                                 ('fc3', nn.Linear(hidden2, 102, bias=True)),
                                 ('output', nn.LogSoftmax(dim=1))
                                 ]))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
        
    # use NLLLoss as criterion
    criterion = nn.NLLLoss()
    
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every=5
    
    print("\n\nTraining.......")
    for epoch in range(epochs):
        # Model in training mode
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                # Model in inference mode
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                      
    return model

                      
def save_checkpoint(model, architecture, modelname, save_dir, image_train, filename='checkpt.pth'):
    model.class_to_idx = image_train.class_to_idx
    filepath = save_dir + filename
                                        
    checkpoint = {'architecture': architecture,
                  'model_name': modelname,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
    torch.save(checkpoint, filepath)
                      
                      
def load_checkpoint(filepath, map_location, architecture):
    checkpoint = torch.load(filepath, map_location)
    
    model = eval("models.{}(pretrained=True)".format(architecture))
    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
                  
                      
def test_fun(model, testloader):
    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():                  
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(logps)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss, accuracy
    
def predict(image_path, model, topk, category_names, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    im = process_image(image_path)
    im = im.unsqueeze_(0)
    
    if device=='cuda':
        im = im.cuda().float()
        model=model.cuda()
    else:
        im = im.cpu().float()
    
    model.eval()
    with torch.no_grad():
        logps = model(im)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(topk) 
        
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[idx] for idx in np.array(top_idx)[0]]
        
        class_to_classname = {key:val for key, val in cat_to_name.items()}
        top_classname = [class_to_classname[idx] for idx in top_class]
        
    results = pd.DataFrame({'ClassIndex': np.array(top_idx)[0],
                            'Class': top_class,
                            'Classname': top_classname,
                            'Probability': np.array(top_p)[0]})
        
    return results

