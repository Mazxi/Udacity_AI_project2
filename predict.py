# importing all modules to be used

import torch
from PIL import Image
from torch import nn
from torch import optim
from workspace_utils import active_session
import torch.nn.functional as F
from torchvision.transforms import functional as Fr
from torch.autograd import variable
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import json


def model_loader(filepath, model_architecture, hidden):
    ''' This function accepts the filepath, model_achitecture, and number of hidden units used in building the saved model and produces a model as the result '''
    if "vgg16" in model_architecture:
        loaded_model = models.vgg16(pretrained = True)
        for param in loaded_model.parameters():
            param.requires_grad = False
        loaded_model.classifier = nn.Sequential(nn.Linear(25088, hidden),
                                                nn.ReLU(),
                                                nn.Dropout(p = 0.4),
                                                nn.Linear(hidden, 2208),
                                                nn.ReLU(),
                                                nn.Linear(2208, 102),
                                                nn.LogSoftmax(dim = 1))

    elif "densenet161" in model_architecture:
        loaded_model = models.densenet161(pretrained = True)
        for param in loaded_model.parameters():
            param.requires_grad = False
        loaded_model.classifier = nn.Sequential(nn.Linear(2208, hidden),
                                                nn.ReLU(),
                                                nn.Dropout(p = 0.4),
                                                nn.Linear(hidden, 2208),
                                                nn.ReLU(),
                                                nn.Linear(2208, 102),
                                                nn.LogSoftmax(dim = 1))

    checkpoint = torch.load(filepath)
    loaded_model.class_to_idx = checkpoint["class_to_idx"]
    loaded_model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optim_state_dict"])
    return loaded_model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
          returns an Numpy array'''

    with Image.open(image) as im:
        scale = (224, 224)
        im = im.resize((256, 256))
        im = Fr.center_crop(im, scale)
        np_image = np.array(im)/255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean)/std
        np_image = np_image.transpose((2, 0, 1))
        
    return np_image


def predict(image_path, model, device, topk):
    ''' This function accepts image path, model, device, topk and returns the topk lass and probabilities '''

    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    logps = model(image_tensor)
    ps = torch.exp(logps)
    top_probs, top_class = ps.topk(topk, dim = 1)
    top_probs, top_class = top_probs.cpu(), top_class.cpu()
    top_probs, top_class = top_probs.detach().numpy().tolist(), top_class.detach().numpy().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in top_class[0]]
    
    return classes, top_probs[0]

def main():

    model_architecture = input("Please enter the enter the model architecture of the model to be loaded: ")
    hidden = int(input("Please enter the number of hidden units used to train the model: "))
    device = input ("Please enter the mode choose between gpu and cpu: ")
    topk = int(input("Please enter number of top probability: "))
    model_path = input("Please enter the checkpoint file path: ")
    json_path = input("Please enter the file path for the json file: ")
    
    if "gpu" in device:
        device = "cuda"
        
    print("loading the saved model...")
    loaded_model = model_loader(model_path, model_architecture, hidden)
    loaded_model.to(device)
    print("...")
    print("model loaded.")
  
    
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    image_path = input("Please enter the image path: ")
    classes, top_probs = predict(image_path, loaded_model, device, topk)

    top_flowers = [cat_to_name[name] for name in classes]

    print("predicting...")
    print("...")

    print(f"This flower is most likely to be {top_flowers[0]} in class {classes[0]} with a probability of {top_probs[0] :.3f}")
    for i in range(1, len(classes)):
        print(f"The flower might also be {top_flowers[i]} in class {classes[i]} with a probability of {top_probs[i] :.3f}")


if __name__ == "__main__":
    main()
