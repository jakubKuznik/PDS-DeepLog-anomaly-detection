# Faculty: BUT FIT 
# Course: PDS
# Project Name: Log Analysis using ML.
# Name: Jakub Kuznik
# Login: xkuzni04
# Year: 2024


# This file contains log parser that will preprocess the 
# log file into the format acceptable for DeepLog NN 

import pandas as pd
import datetime
import re

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
        self.log_file        = self.open_file(log_file)
        self.last_timestamp  = 0
        self.num_logs        = 0 # how many logs are in dataset 

        ## This is table with all the logs  
        # event E1-E29, ti+1 - ti, pid, level={INFO,WARNING,ERROR}, component={dfs.DataNode$PacketResponder} 
        self.all_logs   = pd.DataFrame(columns=['annotation','event', 'time_diff', 'pid', 'level', 'component'])
        
    # Destructor closes the file 
    def __del__(self):
        self.log_file.close()

    # Open file for reading 
    @staticmethod
    def open_file(fileName):
        f = ""
        try:
            f = open(fileName)
        except FileNotFoundError:
            print(f"Error: Log file '{self.log_file}' not found.")
            exit()
        return f

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
        self.num_logs = len(lines)
        
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
        annotation=""

        
        time_diff  = self.__parse_time(parts[0], parts[1])
        pid        = parts[2]
        level      = parts[3]
        component  = parts[4]
        event      = self.__get_event(' '.join(parts[5:]))
        
        for part in parts:
            if part.startswith("blk_"):
                annotation = part
                break

        # Append parsed log into the DataFrame 
        log_entry = {'annotation': annotation, 'event': event, 'time_diff': time_diff, 'pid': pid, 'level': level, 'component': component}
        self.all_logs = self.all_logs._append(log_entry, ignore_index=True)

    # Convert out both given LogParser into the ONE-HOT encode. 
    #  It has to be done on both of then because we need all the possible classes 
    # 
    # Log: 
    # {'event': 'E5', 'time_diff': 0, 'pid': '143', 'level': 'INFO', 'component': 'bbb'}
    # 
    # Atributtes that are converted: 
    #  event, pid, level, component 
    # 
    # @return dimension of the dataset after encoding 
    @staticmethod 
    def one_hot_encoding(training: 'LogParser', testing: 'LogParser'):
        
        # Find the unique values
        merge = pd.concat([training.all_logs, testing.all_logs])
 
        # Get the number of rows in the training data
        num_training_rows = training.all_logs.shape[0]

        event_one_hot       = pd.get_dummies(merge['event'], prefix='event')
        pid_one_hot         = pd.get_dummies(merge['pid'], prefix='pid')
        level_one_hot       = pd.get_dummies(merge['level'], prefix='level')
        component_one_hot   = pd.get_dummies(merge['component'], prefix='component')

        # Do the actuall encoding  
        df_encoded = pd.concat([merge.drop(columns=['event', 'pid', 'level', 'component']),
            event_one_hot, pid_one_hot, level_one_hot, component_one_hot], axis=1)
        
        # Split the encoded data back into training and testing sets
        training.all_logs = df_encoded.iloc[:num_training_rows]
        testing.all_logs = df_encoded.iloc[num_training_rows:]

        return training.all_logs.shape[1]
