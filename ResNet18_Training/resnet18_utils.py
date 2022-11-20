# Imports:
from __future__ import print_function, division
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

import time
import os
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Clear cuda cache before starting:
torch.cuda.empty_cache()
cudnn.benchmark = True

#########################################################################################################################################
def resnet18_data_preprocessing(data_path, img_resize, batch_size=32, workers=2, device='cuda:0'):

    '''
        ResNet18 Torch Dataloarder config:
    '''
    
    # Clear cuda cache before starting:
    torch.cuda.empty_cache()
    cudnn.benchmark = True
       
    # Compose data transformations:
    # Data augmentation and normalization for training.
    # Just normalization for validation.
    # ResNet Normalizarion Values: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        'val': transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Apply transformations to dataset within data_path.
    # Dataset must be a dir with train and val folders, each one with a folder for each class.
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), 
                                              data_transforms[x]) 
                                              for x in ['train', 'val']}
    
    # Add dataset to dataloaders:
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size  = batch_size, 
                                                  shuffle     = True, 
                                                  num_workers = workers)
                                                  for x in ['train', 'val']}
    
    # Get dataset sizes: 
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names   = image_datasets['train'].classes

    # Config device:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    return dataloaders, dataset_sizes, class_names, device

#########################################################################################################################################
def train_model(model, dataloaders, dataset_sizes, device, save_path, criterion, optimizer, scheduler, num_epochs=25):
    
    '''
        Torch training loop:
    '''

    # Start timer:
    since = time.time()
    print('#################################################################################')
    print(f'Start training model:')
    print('#################################################################################')

    # Initial values:
    best_acc       = 0.0
    best_loss      = 20.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loop in number of epochs:
    for epoch in range(num_epochs):
        print('#################################################################################')
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('#################################################################################')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            print(f'Current Phase: {phase}.')
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss     = 0.0
            running_corrects = 0

            # Iterate over data in dataloaders:
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward -> track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs  = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss     = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update loss statistics
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            # Update lr with scheduler:
            if phase == 'train':
                scheduler.step()
                
            ################################################################################################
            # Calculate epoch loss:
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            
            # Iteration Output:
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            #################################################################################################
            # Save model state if current iteration is the best:
            
            # In validation phase: First Criterion - Check if current accuracy is better than previous:
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best yet.')
                print('Saving...')
                    
                # load best model weights:
                model.load_state_dict(best_model_wts)

                # Saving best:
                torch.save(model.state_dict(), f'{save_path}/best_ep{epoch}.pt')
    
            # In validation phase: Second Criterion - Check if current loss is better than previous:
            if phase == 'val' and epoch_acc == best_acc:
                
                if epoch_loss < best_loss:
                    
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Best yet.')
                    print('Saving...')

                    # load best model weights
                    model.load_state_dict(best_model_wts)
                    
                    # Saving best:
                    torch.save(model.state_dict(), f'{save_path}/best_ep{epoch}.pt')

    #################################################################################################
    # Training Iteration loop Completed: 
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights and return:
    model.load_state_dict(best_model_wts)
    
    return model

#########################################################################################################################################
def load_ResNet18(model_path, classes, input_size, device):

    '''
        Load ResNet18 Torch model from state dict saved in model_path:
    
    '''

    # Get torch ResNet18 config:
    model    = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model    = model.to(device)

    # Load state dict and put in eval mode:
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Build ResNet18 pre_processing compose:
    pre_process = transforms.Compose([transforms.Resize((input_size, input_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    return model, pre_process

#########################################################################################################################################
def ResNet18_inference(img_path, model, classes, pre_process, batch_size, device):

    '''
        ResNet18 Torch Inference:
    '''

    # Loading img with PIL:
    img = Image.open(img_path)

    # Start timer:
    start = time.time()

    # Preprocessing img to tensor:
    image_tensor = pre_process(img)
    input_tensor = image_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Adjust input with batch size:
    input_batch = input_tensor.repeat(batch_size, 1, 1, 1).to(device)

    # Run inference and get detected class:
    outputs   = model(input_batch)
    class_det = classes[np.argmax(outputs[0].cpu().detach().numpy())]

    # Print outputs:
    end = time.time()
    print('Process Time:', (end-start)*1000, 'ms')
    print('Output:', outputs[0])
    print('Class Detected:', class_det)

    return class_det, outputs

#########################################################################################################################################


    
