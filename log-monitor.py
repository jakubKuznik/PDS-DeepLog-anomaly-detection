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


def help():
    print("log-monitor")
    print("     -training <file>")
    print("     -testing <file>")
    print("     -<params> par1=val1,par2=val2,...")

def my_errors(code):
    if code == 0: 
        print("Error: Wrong arguments try: ", file=sys.stderr)
        help()
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
                my_errors(0)
        elif argv[i] == "-testing":
            i += 1 
            if i < len(argv):
                testing_file = argv[i]
            else: 
                my_errors(0)
        elif argv[i] == "-params":
            i += 1 
            if i < len(argv):
                params = argv[i]
            else: 
                my_errors(0)
        i += 1

    if training_file == "" or testing_file == "":
        my_errors(0)

    return training_file, testing_file, params

# Open file for reading 
def open_file(fileName):
    f = ""
    try:
        f = open(fileName)
    except FileNotFoundError:
        print(f"Error: Log file '{self.log_file}' not found.")
        exit()
    return f


## This class works ass class for log parsing 
# EventId,EventTemplate
# E1,[*]Adding an already existing block[*]
# E2,[*]Verification succeeded for[*]
# E3,[*]Served block[*]to[*]
# E4,[*]Got exception while serving[*]to[*]
# E5,[*]Receiving block[*]src:[*]dest:[*]
# E6,[*]Received block[*]src:[*]dest:[*]of size[*]
# E7,[*]writeBlock[*]received exception[*]
# E8,[*]PacketResponder[*]for block[*]Interrupted[*]
# E9,[*]Received block[*]of size[*]from[*]
# E10,[*]PacketResponder[*]Exception[*]
# E11,[*]PacketResponder[*]for block[*]terminating[*]
# E12,[*]:Exception writing block[*]to mirror[*]
# E13,[*]Receiving empty packet for block[*]
# E14,[*]Exception in receiveBlock for block[*]
# E15,[*]Changing block file offset of block[*]from[*]to[*]meta file offset to[*]
# E16,[*]:Transmitted block[*]to[*]
# E17,[*]:Failed to transfer[*]to[*]got[*]
# E18,[*]Starting thread to transfer block[*]to[*]
# E19,[*]Reopen Block[*]
# E20,[*]Unexpected error trying to delete block[*]BlockInfo not found in volumeMap[*]
# E21,[*]Deleting block[*]file[*]
# E22,[*]BLOCK* NameSystem[*]allocateBlock:[*]
# E23,[*]BLOCK* NameSystem[*]delete:[*]is added to invalidSet of[*]
# E24,[*]BLOCK* Removing block[*]from neededReplications as it does not belong to any file[*]
# E25,[*]BLOCK* ask[*]to replicate[*]to[*]
# E26,[*]BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]
# E27,[*]BLOCK* NameSystem[*]addStoredBlock: Redundant addStoredBlock request received for[*]on[*]size[*]
# E28,[*]BLOCK* NameSystem[*]addStoredBlock: addStoredBlock request received for[*]on[*]size[*]But it does not belong to any file[*]
# E29,[*]PendingReplicationMonitor timed out block[*]

## Structured log 
# LineId	Date	Time	Pid	Level	Component	Content	EventId	EventTemplate
#   1	    081109	203615	148	INFO	dfs.DataNode$PacketResponder	PacketResponder 1 for block blk_38865049064139660 terminating	E10	PacketResponder <*> for block blk_<*> terminating
## Real log 
#   081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating

class LogParser:
    def __init__(self, log_file):
        self.log_file = open_file(log_file)

    # Destructor closes the file 
    def __del__(self):
        self.log_file.close()

    def parse_line(self):
        line = self.log_file.readline() 
        if line: # check if we are on EOF 
            return line.strip()  # Strip to remove leading/trailing whitespace
        else:
            return None


def main():
    
    training_file, testing_file, params = parse_arguments(sys.argv)

    log_parser = LogParser(training_file)
    parsed_log = log_parser.parse_line()
    while True:
        line = log_parser.parse_line()
        if line == None:
            break
        else:
            print(line)

if __name__ == "__main__":
    main()