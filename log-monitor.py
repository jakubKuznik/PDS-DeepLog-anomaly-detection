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
import datetime
import pandas as pd
import re


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

    # It is optimalized by library 
    patterns = {
        'E1': re.compile(r'.*Adding an already existing block.*'),
        'E2': re.compile(r'.*Verification succeeded for.*'),
        'E3': re.compile(r'.*Served block.*to.*'),
        'E4': re.compile(r'.*Got exception while serving.*to.*'),
        'E5': re.compile(r'.*Receiving block.*src:.*dest:.*'),
        'E6': re.compile(r'.*Received block.*src:.*dest:.*of size.*'),
        'E7': re.compile(r'.*writeBlock.*received exception.*'),
        'E8': re.compile(r'.*PacketResponder.*Exception.*'),
        'E9': re.compile(r'.*Received block.*of size.*from.*'),
        'E10': re.compile(r'.*PacketResponder.*Exception.*'),
        'E11': re.compile(r'.*PacketResponder.*for block.*terminating*'),
        'E12': re.compile(r'.*:Exception writing block.*to mirror.*'),
        'E13': re.compile(r'.*Receiving empty packet for block.*'),
        'E14': re.compile(r'.*Exception in receiveBlock for block.*'),
        'E15': re.compile(r'.*Changing block file offset of block.*from.*to.*meta file offset to.*'),
        'E16': re.compile(r'.*:Transmitted block.*to.*'),
        'E17': re.compile(r'.*:Failed to transfer.*to.*got.*'),
        'E18': re.compile(r'.*Starting thread to transfer block.*to.*'),
        'E19': re.compile(r'.*Reopen Block.*'),
        'E20': re.compile(r'.*Unexpected error trying to delete block.*BlockInfo not found in volumeMap.*'),
        'E21': re.compile(r'.*Deleting block.*file.*'),
        'E22': re.compile(r'.*BLOCK\* NameSystem.*allocateBlock:.*'),
        'E23': re.compile(r'.*BLOCK\* NameSystem.*delete:.*is added to invalidSet of.*'),
        'E24': re.compile(r'.*BLOCK\* Removing block.*from neededReplications as it does not belong to any file.*'),
        'E25': re.compile(r'.*BLOCK\* ask.*to replicate.*to.*'),
        'E26': re.compile(r'.*BLOCK\* NameSystem.*addStoredBlock: blockMap updated:.*is added to.*size.*'),
        'E27': re.compile(r'.*BLOCK\* NameSystem.*addStoredBlock: Redundant addStoredBlock request received for.*on.*size.*'),
        'E28': re.compile(r'.*BLOCK\* NameSystem.*addStoredBlock: addStoredBlock request received for.*on.*size.*But it does not belong to any file.*'),
        'E29': re.compile(r'.*PengingReplicationMonitor timed out block.*'),
    }
    
    def __init__(self, log_file):
        self.log_file        = open_file(log_file)
        # todo maybe there will be problem with first timestamp 
        self.last_timestamp  = 0

        ## This is table with all the logs  
        # event E1-E29, ti+1 - ti, pid, level={INFO,WARNING,ERROR}, component={dfs.DataNode$PacketResponder} 
        self.all_logs = pd.DataFrame(columns=['event', 'time_diff', 'pid', 'level', 'component'])


    # Destructor closes the file 
    def __del__(self):
        self.log_file.close()

    # 081109 -> 2008.11.09
    # 203615 -> 20:36:15
    # First create Current Timestamp and then 
    #  @return last_timestamp - current_timestamp 
    def __parse_time(self, date, time):
        date = datetime.datetime.strptime(date, "%y%m%d")
        time = datetime.datetime.strptime(time, "%H%M%S")
        
        # Combine date and time
        timestamp = datetime.datetime.combine(date.date(), time.time())
        unix_timestamp = int(timestamp.timestamp())
        
        if self.last_timestamp == 0:
            time_diff = 0
        else: 
            time_diff = last_timestamp - timestamp
        
        last_timestamp = timestamp
        
        return time_diff
    ##
    # Get event from string part of the log using pattern 
    def __get_event(self, str):
        matched = ""

        # Go throught patterns, if not match retun empty 
        for pattern, compiled_pattern in self.patterns.items():
            if compiled_pattern.match(str):
                return pattern
        my_errors(1, str)
        
        return matched 
    
    # Read whole file at once and call parse_line() on each line 
    def parse_file(self):
        lines = self.log_file.readlines()
        i = 0 
        for l in lines:
            i += 1 
            if i % 1000 == 0:
                print(i)
            self.parse_line(l)

    ##
    # Parse one line of the log file into the dataframe 
    def parse_line(self, line):
        parts = line.split()  

        event=""
        time_diff=0
        pid=""
        level=""
        component=""

        # check if we are on EOF 
        if not line: 
            return None
        
        
        time_diff = self.__parse_time(parts[0], parts[1])
        pid       = parts[2]
        level     = parts[3]
        component = parts[4]
        event     = self.__get_event(' '.join(parts[5:]))

        log_entry = {'event': event, 'time_diff': time_diff, 'pid': pid, 'level': level, 'component': component}
        self.all_logs = self.all_logs._append(log_entry, ignore_index=True)


        return line.strip()  # Strip to remove leading/trailing whitespace


def main():
    
    training_file, testing_file, params = parse_arguments(sys.argv)

    # Init log_parser with training_file 
    log_parser = LogParser(training_file)
    
    log_parser.parse_file()
    
    print(log_parser.all_logs)

if __name__ == "__main__":
    main()