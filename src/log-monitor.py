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
    total_loss = 0.0 
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    for batch_data, batch_labels in data_loader:
        D, L = batch_data.to(device), batch_labels.to(device)
        # Pass the batch_data through the model
        # output is 64 x ([2])
        # [0.4123, 0.3214] normal probability and annomaly probability
        output = model(D)
        # Compute the binary cross-entropy loss
        loss = loss_func(output, L)  

        # Backpropagation
        loss.backward()
        
        optimizer.step()
        
        optimizer.zero_grad()

        total_loss += loss.item()

    # Calculate average loss
    average_loss = total_loss / num_batches
    print("Training Loss:", average_loss)



def test(dataloader, model, loss_fn):
    # size of dataset 
    size = len(dataloader.dataset)
    ## how many batches are in the dataset 
    num_batches = len(dataloader)
    ## same as we did model.train() 
    ##  This set model into the evaluation state 
    model.eval()
    test_loss, correct = 0, 0
    ## with torch.no_grad() disable gradient callculation for a testing 
    ##   so it will be faster 
        
    total_correct_normal = 0
    total_correct_anomaly = 0
    total_wrong_normal = 0
    total_wrong_anomaly = 0
    
    with torch.no_grad():
    
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
        
            # Convert probabilities to binary predictions
            predicted = (pred > 0.5).float()  # Threshold at 0.5
        
            # Count correct and wrong predictions
            correct_normal = ((predicted == 0) & (y == 0)).sum().item()
            correct_anomaly = ((predicted == 1) & (y == 1)).sum().item()
            wrong_normal = ((predicted == 1) & (y == 0)).sum().item()
            wrong_anomaly = ((predicted == 0) & (y == 1)).sum().item()
        
            # Tally the counts
            total_correct_normal += correct_normal
            total_correct_anomaly += correct_anomaly
            total_wrong_normal += wrong_normal
            total_wrong_anomaly += wrong_anomaly
    
    accuracy_all = ((total_correct_normal + total_correct_anomaly) / size) * 100
    accuracy_normal = (total_correct_normal / (total_correct_normal + total_wrong_normal)) * 100
    accuracy_anomaly = (total_correct_anomaly / (total_correct_anomaly + total_wrong_anomaly)) * 100

    # Print accuracy percentages
    print("")
    print("Overall Accuracy: {:.2f}%".format(accuracy_all))
    print("Normal Accuracy: {:.2f}%".format(accuracy_normal))
    print("Anomaly Accuracy: {:.2f}%".format(accuracy_anomaly))
    
    print("Correct normal predictions:", total_correct_normal)
    print("Wrong normal predictions:", total_wrong_normal)
    print("Correct anomaly predictions:", total_correct_anomaly)
    print("Wrong anomaly predictions:", total_wrong_anomaly)
    


def main():
    
    batch_size        = 64
    hidden_features   = 64 
    LSTM_layers       = 2 
    output_size       = 2
    epochs            = 1000
    # [1,2,3] [2,3,4] [3,4,5]
    window_size       = 10
    imbalance_factor = 5
    
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
    # In preprocessor thre will be the dataset 
    preproces_train = Preproces(parser_train.all_logs, window_size)
    preproces_test  = Preproces(parser_test.all_logs, window_size)

    train_input_features    = preproces_train.features
    test_input_features     = preproces_test.features
    
    train_data_loader   = DataLoader(preproces_train.dataset, batch_size=batch_size, shuffle=False, drop_last=True) 
    test_data_loader    = DataLoader(preproces_test.dataset, batch_size=batch_size, shuffle=False, drop_last=True) 
    
    # Instantiate the model
    # Input for LSTM 
    # (batch-size, timestamp, input_dim) 
    model = DeepLog(train_input_features, LSTM_layers, hidden_features, output_size, batch_size)
    model.to(device)

    
    # ANNOMALY 
    # [1,0]
    class_weights = torch.tensor([imbalance_factor, 1]).to(device)

    # Instantiate the loss function with class weights
    # CrossEntropyLoss automatically applies weights to each class
    loss = nn.BCEWithLogitsLoss() 

    # TODO maybe add constant like 1r=1e-3
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for ep in range(1, epochs+1):
        
        print("--------------------------------------")
        print("ep: ", ep, "total epochs:", epochs)

        train(train_data_loader, model, loss, optimizer)
        
        if ep % 5 == 0:         
            test(test_data_loader, model, loss) 
        print("--------------------------------------")
        

if __name__ == "__main__":
    main()