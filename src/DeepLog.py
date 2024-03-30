# Faculty: BUT FIT 
# Course: PDS
# Project Name: Log Analysis using ML.
# Name: Jakub Kuznik
# Login: xkuzni04
# Year: 2024


## This file contains architecture of DeepLog NN model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

## Init GPU 
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f"Using {device} device")


class DeepLog(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x 