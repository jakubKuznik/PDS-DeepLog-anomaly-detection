# Deep Learning for Log File Anomaly detection using DeepLog
## Faculty: BUT FIT, Course: PDS

Name: Jakub Kuznik  
Login: xkuzni04  

# Points 
*/25

## Purpose of Files  
| File        | Purpose |
|-------------|---------|
| src/*   | All the source codes |
| logs/*  | All the logs | 
| README.md | Readme with all the instructions | 
| requirements.txt | required python libraries | 
| xkuzni04.pdf | Technical documentation of the project |
| src/log-monitor.py | Main source code | 
| src/DeerLog.py | DeepLog model implementation in Pytorch | 
| src/parser.py | File for parsing raw log data | 
| src/annotation/* | Script for data anotation |   
    
## Instal Dependencies
```pip3 install -r requirements.txt```  

## Execution
Measured evaluation:   
```python3 src/log-monitor.py -training logs/test-1-test.log -testing logs/test-1-train.log```   

Usefull commands:   
```python3 src/log-monitor.py -training <(head -n 50000 logs/HDFS-annotate.log) -testing <(tail -n +50001 logs/HDFS-annotate.log)```  
```python3 src/log-monitor.py -training <(head -n 50000 logs/HDFS-annotate.log) -testing <(tail -n +50001 logs/HDFS-annotate.log | head -n 50000)```  


### Setting up Python Environment  
- Create a Python virtual environment in the current folder:  
   ```python -m venv .``` 
- Activate the venv:  
   ```  source bin/activate```   
- Install the required packate into the venv:  
   ```pip3 install package```  
- Deactivate venv:   
   ```deactivate```  
- Create requirements.txt file:   
   ```python -m pip freeze > requirements.txt``` 
- Install from requirements  
    ```pip install -r requirements.txt``` 

## Sources 
DeepLog baseline: 
Du, M., Li, F., Zheng, G. and Srikumar, V. DeepLog: Anomaly Detection and  
Diagnosis from System Logs through Deep Learning. In:. October 2017, p. 1285–1298.  
DOI: 10.1145/3133956.3134015. ISBN 978-1-4503-4946-8   

Used log dataset: HDFS from: https://github.com/logpai/loghub/tree/master/HDFS by:   
Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael Jordan.  
Detecting Large-Scale System Problems by Mining Console Logs, in Proc.   
of the 22nd ACM Symposium on Operating Systems Principles (SOSP), 2009.  
Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. Loghub:   
A Large Collection of System Log Datasets for AI-driven Log Analytics.  
IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.  

