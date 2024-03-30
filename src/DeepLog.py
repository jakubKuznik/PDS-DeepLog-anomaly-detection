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
import torch.nn.functional as Fun



## Init GPU 
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f"Using {device} device")


## If we want to create pytorch NN we need to inherit from nn.module 
class DeepLog(nn.Module):

    # Constructor that defines the NN 
    # input_features - how many dimensions will be in data 
    # LSTM_layers    - how many layers of LSTM we want 
    # LSTM_hidden_layers - How many features will be in hidden layer which 
    #     represents the memory of the NN 
    # ouput_size 
    def __init__(self, input_features, LSTM_layers, LSTM_hidden_features, output_size):
        super().__init__()

        # TODO i assume that the hidden_features and input features will always be the same 

        self.input_features         = input_features
        self.LSTM_layers            = LSTM_layers
        self.LSTM_hidden_features   = LSTM_hidden_features
        self.output_size            = output_size

        # -> [BATCH, ]

        # Multi-layer LSTM - long short-term memory RNN (Recurent NN)
        self.lstm   = nn.LSTM(input_features, LSTM_hidden_features, LSTM_layers, batch_first=True)
        # fully connected linear layer 
        # It is great for finding some complex patterns in Data and reduce dimension
        # nn.Linear(512, 10) -> reduce dim from 512 -> 10 
        self.linear     = nn.Linear(LSTM_hidden_features, output_size)
        # Soft max will be applied to the last dimension of input tensor (-1)
        #  We are using softmax because we want a probability of anomaly 
        self.softmax    = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        return x

