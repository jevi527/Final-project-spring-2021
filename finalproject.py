import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time

def accuracy(predicted, true):
    return (torch.argmax(predicted, dim=1)==true).float().mean() * 100

path = "/Downloads/archive/brain_tumor_dataset"
batch_size = 64
data_image = {x:datasets.ImageFolder(root = os.path.join(path,x))
              for x in ["no", "yes"]}

data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=batch_size,
                                                    shuffle = True)
                     for x in ["no", "yes"]}
model = models.vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad = True

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4))
model.to(device)

cost = torch.nn.CrossEntropyLoss() # cost funtion

optimizer = torch.optim.Adam (model.classifier.parameters(),lr=0.01)



