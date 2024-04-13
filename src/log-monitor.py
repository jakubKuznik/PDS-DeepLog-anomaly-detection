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


def main():
    
    training_file, testing_file, params = parse_arguments(sys.argv)

    # Init log_parser with training_file and testing file 
    parser_train = LogParser(training_file)
    parser_test  = LogParser(testing_file)

    # Parse the input files  
    parser_train.parse_file()
    parser_test.parse_file()
    
    print(parser_train.all_logs)
    print(parser_test.all_logs)

    # Encode logs using one hot encoding. log_parser.all_logs
    #    will contain encoded logs after this 
    # Carefull the training dataset idealy should have all the possible values
    # TODO maybe delete component or PID if the training takes to long  
    # dimenstion = Rows x Columns  
    # TODO rewrite it so we will get all the features both from training and 
    #   testing files and then do one_hot_encoding() 
    dimension = LogParser.one_hot_encoding(parser_train, parser_test)
    
    print(parser_train.all_logs)
    print(dimension) 


    ## preapre
    preproces_train = Preproces(parser_train.all_logs)
    
    # ([715, 93])
    print(preproces_train.tensor_labeled['data'])
    print(preproces_train.tensor_labeled['labels'])

    input_features    = preproces_train.features 
    batch_size        = 64
    # window_size       = 64
    hidden_features   = 64 
    LSTM_layers       = 2 
    output_size       = 2
    epochs            = 50 

    print(preproces_train.tensor_labeled['data'].shape)
    print(preproces_train.tensor_labeled['labels'].shape)
    
    print(preproces_train.tensor_labeled['data'])
    print(preproces_train.tensor_labeled['labels'])
    dataset = TensorDataset(preproces_train.tensor_labeled['data'], 
                            preproces_train.tensor_labeled['labels'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
    
    # Instantiate the model
    model = DeepLog(input_features, LSTM_layers, hidden_features, output_size)
    print(model)
    model.to(device)


if __name__ == "__main__":
    main()