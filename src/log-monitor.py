# Faculty: BUT FIT 
# Course: PDS
# Project Name: Log Analysis using ML.
# Name: Jakub Kuznik
# Login: xkuzni04
# Year: 2024

# Execution 
# log-monitor -training <file> -testing <file> -<params> 
#   -training <file>; a data set used to train the model
#   -testing <file>: a data set used for testing the classification 
#   -<params>: a list of parametes required for the specific model 
#    (threshold, time window) etc. (TBD) in format par1=val1, par2=val2,...

# log-monitor -training logs/HDFS_v1/HDFS.log -testing logs/HDFS_v1/HDFS.log

import os.path
import time
import sys

from parser import LogParser
from DeepLog import DeepLog, device, Preproces
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn 
import torch.optim as optim 

def help():
    print("log-monitor")
    print("     -training <file>")
    print("     -testing <file>")
    print("     -<params> par1=val1,par2=val2,...")

def my_errors(code, str):
    if code == 0: 
        print("Error: Wrong arguments try: ", file=sys.stderr)
        help()
        exit()
    elif code == 1:
        print("Unknown log format: " + str, file=sys.stderr)
        exit()

# @return training_file, testing_file, params 
def parse_arguments(argv):
    
    training_file = ""    
    testing_file  = ""    
    params = ""    
    i = 0

    while i < len(argv): 

        if argv[i] == "-h" or argv[i] == "--help" or argv[i] == "-help":
            help()
        elif argv[i] == "-training":
            i += 1 
            if i < len(argv):
                training_file = argv[i]
            else: 
                my_errors(0, "")
        elif argv[i] == "-testing":
            i += 1 
            if i < len(argv):
                testing_file = argv[i]
            else: 
                my_errors(0, "")
        elif argv[i] == "-params":
            i += 1 
            if i < len(argv):
                params = argv[i]
            else: 
                my_errors(0, "")
        i += 1

    if training_file == "" or testing_file == "":
        my_errors(0, "")

    return training_file, testing_file, params

# train one epoch
def train(data_loader, model, loss_func, optimizer):
        
    model.train()
    # Iterate over the data loader
    for batch_data, batch_labels in data_loader:

        D, L = batch_data.to(device), batch_labels.to(device)

        # Pass the batch_data through the model
        # output is 64 x ([2])
        # [0.4123, 0.3214] normal probability and annomaly probability
        output = model(D)

        # Compute the binary cross-entropy loss
        loss = loss_func(output, L)  # Squeeze the output to remove extra dimensions

        # Backpropagation
        loss.backward()
        
        optimizer.step()
        
        optimizer.zero_grad()



# evaluate model on testing data
def test():
    model.eval()
    print("test")


def main():
    
    training_file, testing_file, params = parse_arguments(sys.argv)

    # Init log_parser with training_file and testing file 
    parser_train = LogParser(training_file)
    parser_test  = LogParser(testing_file)

    # Parse the input files  
    parser_train.parse_file()
    parser_test.parse_file()
    
    # Encode logs using one hot encoding. log_parser.all_logs
    #    will contain encoded logs after this 
    # Carefull the training dataset idealy should have all the possible values
    # TODO maybe delete component or PID if the training takes to long  
    # dimenstion = Rows x Columns  
    # TODO rewrite it so we will get all the features both from training and 
    #   testing files and then do one_hot_encoding() 
    dimension = LogParser.one_hot_encoding(parser_train, parser_test)

    ## preapre
    window_size       = 10
    # In preprocessor thre will be the dataset 
    preproces_train = Preproces(parser_train.all_logs, window_size)

    input_features    = preproces_train.features
    batch_size        = 64
    hidden_features   = 64 
    LSTM_layers       = 2 
    output_size       = 2
    epochs            = 50 
    
    data_loader = DataLoader(preproces_train.dataset, batch_size=batch_size, shuffle=False, drop_last=True) 
    
    # Instantiate the model
    # Input for LSTM 
    # (batch-size, timestamp, input_dim) 
    model = DeepLog(input_features, LSTM_layers, hidden_features, output_size, batch_size)
    model.to(device)

     
    loss = nn.CrossEntropyLoss()
    # todo maybe add constant like 1r=1e-3
    optimizer = optim.Adam(model.parameters())

    for ep in range(1, epochs+1):

        train(data_loader, model, loss, optimizer)
        
        
        print("ep: ", ep, "total epochs:", epochs)

if __name__ == "__main__":
    main()