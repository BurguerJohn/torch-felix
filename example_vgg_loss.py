#VGG LOSS
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from felix import *

device = "cpu"
model = torchvision.models.vgg19(True).to(device)

#Create a hook for all Conv2D modules
hook_instance = nn.Conv2d

transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5])
            #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

weight_ini = 1.0
weight_end = 1.0
curve = 1
verbose = True

extractor = FelixExtractor(model, hook_instance, transf, weight_ini, weight_end, curve, verbose)
loss = FelixLoss(extractor)


t1 = torch.rand(1, 3, 224, 224)
t2 = torch.rand(1, 3, 224, 224)

print("Loss of the tensor with itself should return 0.")
print(loss(t1, t1))

print("Correct way to get the loss")
print(loss(t1, t2))

