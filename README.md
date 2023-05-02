# torch-felix: Pytorch Feature Extraction with Loss Integration eXtractor


Torch-Felix is a flexive feature extraction code designed to work with the majority
of pre-trained PyTorch models. The extracted features can be used as a loss function for various training and optimization tasks. 

## Features
- Use with most pre-trained models, with little code edit required.
- Easy to setup weights per layer.
- Simple code, easy to undestand.

## Installation
Copy the file **felix.py** to your project root folder.

## Usage
Felix code will hook all the layers of a selected instance of a model and apply a weight for it.
To undestand the weights, look the file *example_weight_curve_visualization.py* and the images on the *img* folder.

You can check the examples code for more usages, here is the code to check the feature loss of the VGG19 network:
```python
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
```

## The logic in this project:
This project is not affiliated with any scientific paper; it was created through trial and error while I used it in my personal projects. I would like to explain some choices I made and the default configurations for this project:

- I have only had success using the L1 loss; when I use the L2 loss, it rarely gives me satisfactory results.
- For my projects, using a weight of 1.0 for all layers usually provides the best results, but this depends on your project and goals.
- Logically, you would want to use the hook after the activation functions; however, for my projects, it usually yields better results before calling the activation functions.
- Although most models are trained with normalization `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)`, my projects tend to produce better results when I use `mean=[0.5], std=[0.5]`.
- In some models, using the activation `nn.ReLU(True)` after the hook can cause backpropagation issues, so the code replaces these functions with `nn.ReLU(False)`.
- Depending on the model, you may want to remove the last layer before sending it to the FelixExtractor, as the final layer is often a specific result that the model was trained to generate.

## License
Torch-Felix is licensed under the MIT License. 

