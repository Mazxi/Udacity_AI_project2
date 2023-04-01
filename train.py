# importing all functions that will be needed
import torch
from PIL import Image
from torch import nn
from torch import optim
from workspace_utils import active_session
import torch.nn.functional as F
from torchvision.transforms import functional as Fr
from torch.autograd import variable
from torchvision import datasets, transforms, models
import numpy as np

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print("loading and preprocessing data for model building...")
# Defining transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(30),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])
                                                    ]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])
                                                    ]),
                       'test': transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])
                                                   ])}
# Loading the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
                      }
# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64)
                   }

print("... processing done!")
print("...")
print("Model training about to commence")

def train_network(model, criterion, optimizer, epochs = 1, device = "cuda"):
    ''' This function takes in model, criterion, optimizer, epochs, device and
    a trained model as result '''
    
    # defining and setting step, running loss and print every to their default values
    step = 1
    running_loss = 0
    print_every = 50
    # start of the loop, it loops through for the number of epochs specified
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloaders['train']:
            # incrementing step by 1
            step += 1
            # saving the images and the labels to device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # making predictions
            logps = model.forward(images)
            # comparing prediction with the actual result
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        # comparing the results
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        # getting the top probability and class
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        # calculating the accuracy of the prediction
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                # printing out the result of the prediction      
                print(f"Epoch {epoch+1}/{epochs}... "
                      f"Train loss: {running_loss/len(dataloaders['train']):.3f}... "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}... "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")                                                                  
                running_loss = 0
                model.train()
                                                         
    return model

def save_checkpoint(model, model_name, optimizer):
    ''' This model takes in the model, model_name, and optimizer and prints out the path of the saved model '''
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    if "vgg" in model_name:
        parameters = {
            "class_to_idx": model.class_to_idx,
            "classifier": model.classifier,
            "state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "optimizer": optimizer
            }

    elif "densenet" in model_name:
        parameters = {
            "class_to_idx": model.class_to_idx,
            "classifier": model.classifier,
            "state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "optimizer": optimizer
            }
    path_name = "checkpoint1.pth"
    torch.save(parameters, path_name)
    print(f"the path of the saved model is '{path_name}'")
    
def main():

    # taking inputs from user
    model_architecture = input("Please choose between the densenet161 or vgg16: ")
    learning_rate = float(input("Please enter the learning rate: "))
    epochs = int(input("Please enter the number of epochs: "))
    device = input("Please enter the mode to train on either gpu or cpu: ")
    hidden = int(input(" Please enter the number of hidden units please it shouldn't be more than 2208 units: "))

    if "gpu" in device:
        # saving gpu as cuda
        device = "cuda"
    
    if "vgg16" in model_architecture: # checking if model architecture is vgg16
        model = models.vgg16(pretrained = True)
        # freezing model parameters
        for param in model.parameters():
            param.requires_grad = False
        # creating a new feedforward network
        model.classifier = nn.Sequential(nn.Linear(25088, hidden),
                                         nn.ReLU(),
                                         nn.Dropout(p = 0.4),
                                         nn.Linear(hidden, 2208),
                                         nn.ReLU(),
                                         nn.Linear(2208, 102),
                                         nn.LogSoftmax(dim = 1))

    elif "densenet161" in model_architecture:
        model = models.densenet161(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(2208, hidden),
                                         nn.ReLU(),
                                         nn.Dropout(p = 0.4),
                                         nn.Linear(hidden, 2208),
                                         nn.ReLU(),
                                         nn.Linear(2208, 102),
                                         nn.LogSoftmax(dim = 1))
    # setting the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device);

    # training session
    with active_session():
        train_network(model, criterion, optimizer, epochs, device)

    print("Model training done.")
    print("saving model...")
    save_checkpoint(model, model_architecture, optimizer)
    print("saving done!")

if __name__ == "__main__":
    main()