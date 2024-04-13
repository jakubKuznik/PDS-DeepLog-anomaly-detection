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
import pandas as pd
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


## Init GPU 
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)
print(f"Using {device} device")


# https://github.com/nailo2c/deeplog/blob/master/deeplog/deeplog.py


## If we want to create pytorch NN we need to inherit from nn.module 
class DeepLog(nn.Module):

    # Constructor that defines the NN 
    # input_features - how many dimensions will be in data 
    # LSTM_layers    - how many layers of LSTM we want 
    # LSTM_hidden_layers - How many features will be in hidden layer which 
    #     represents the memory of the NN 
    # ouput_size 
    def __init__(self, input_features, LSTM_layers, LSTM_hidden_features, output_size, batch_size):
      super().__init__()

      # TODO i assume that the hidden_features and input features will always be the same 

      self.input_features         = input_features
      self.LSTM_layers            = LSTM_layers
      self.LSTM_hidden_features   = LSTM_hidden_features
      self.output_size            = output_size


      # -> [BATCH, ]
      self.batch_size = batch_size

      # Multi-layer LSTM - long short-term memory RNN (Recurent NN)
      self.lstm   = nn.LSTM(input_features, LSTM_hidden_features, LSTM_layers, batch_first=True)
      # fully connected linear layer 
      # It is great for finding some complex patterns in Data and reduce dimension
      # nn.Linear(512, 10) -> reduce dim from 512 -> 10 
      self.linear     = nn.Linear(LSTM_hidden_features, output_size)
      # Soft max will be applied to the last dimension of input tensor (-1)
      #  We are using softmax because we want a probability of anomaly 
      
      self.softmax    = nn.LogSoftmax(dim=-1)


    def forward(self, input):

      h0      = torch.zeros(self.LSTM_layers, input.size(0), self.LSTM_hidden_features).to(input.device)
      c0      = torch.zeros(self.LSTM_layers, input.size(0), self.LSTM_hidden_features).to(input.device)
      out, _  = self.lstm(input, (h0, c0))
      out     = self.linear(out[:, -1, :])
      # out = F.log_softmax(out, dim=-1)  # Using F.log_softmax instead of nn.LogSoftmax
      return torch.exp(out)  # Exponentiating to get probabilities




# prepare data from DF to Tensor for NN 
class Preproces():

  # from pandas encoded in ONE-HOT-encoding
  #   prepare annotated tensor
  def __init__(self, dataF, window_size):


    data  =  dataF.iloc[:, 2:].values.astype(float) 
    self.samples, self.features = data.shape
    # 1.0 mean anomaly 
    labels = dataF['annotation'].map({'blk_Normal': False, 'blk_Anomaly': True })


    # Create sequences
    # [1,2,3], [2,3,4], [3,4,5]
    sequences = []
    outputs = []
    for i in range(self.samples - window_size + 1):
      
      sequence = data[i:i+window_size]  # Extract a sequence of data points
      # check if there is an anomaly
      if any(labels[i:i+window_size]):
        outputs.append([True,False])
      else:
        outputs.append([False,True])
      sequences.append(sequence)

    self.data = np.array(sequences)
    self.labels = np.array(outputs)
    
    self.torch_data = torch.tensor(self.data, dtype=torch.float)
    self.torch_labels = torch.tensor(self.labels, dtype=torch.float)
    
    self.dataset = TensorDataset(self.torch_data, self.torch_labels)
