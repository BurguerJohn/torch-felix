import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms


def CreateWeights(start_weight, end_weight, count, curve_weight):
  weights = np.linspace(start_weight, end_weight, count)
  for i in range(len(weights)):
    weights[i] = math.pow(weights[i], curve_weight)
  return weights

class FelixExtractor(torch.nn.Module):
    def __init__(self, model, hook_instance, transforms, start_weight = 1.0, end_weight = 1., curve_weight=1, verbose=True):
        super().__init__()

        if curve_weight < 1:
          raise Exception("Curve can't be smaller than 1")


        self.activations = []
        def getActivation():
            def hook(model, input, output):
                self.activations.append(output)
            return hook
        
        if verbose:
          print("Loading model")

        self.model = model.eval()

        count = 0
        def traverse_modules(module):
            nonlocal count, verbose
            for name, sub_module in module.named_children():
                #full_name = parent_name + '.' + name if parent_name else name
                if isinstance(sub_module, hook_instance):
                    count += 1
                    if verbose:
                      print(f"-> {sub_module}")
                    sub_module.register_forward_hook(getActivation())
                elif isinstance(sub_module, nn.ReLU):
                  setattr(module, name, nn.ReLU(False))
                else:
                    traverse_modules(sub_module)
        
        traverse_modules(self.model)
        if verbose:
          print(f"Total Layers: {count}")
        
        self.weights = CreateWeights(start_weight, end_weight, count, curve_weight)
        self.transforms = transforms
        
    def forward(self, X):
        X = self.transforms(X)
        self.activations = []
        self.model(X)
        return self.activations



class FelixLoss(torch.nn.Module):
  def __init__(self, extractor, loss = nn.L1Loss()):
    super().__init__()
    self.extractor = extractor
    self.loss = loss

  def forward(self, X, Y):
        X = self.extractor(X)
        Y = self.extractor(Y)
        loss = 0
        for i in range(len(X)):
          loss += self.loss(X[i], Y[i].detach()) * self.extractor.weights[i]

        return loss
  
