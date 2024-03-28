# Network Logs Anomaly Detection Using ML
## Faculty: BUT FIT, Course: PDS

Name: Jakub Kuznik  
Login: xkuzni04  

## Overleaf 


## Purpose of Files  
| File        | Purpose |
|-------------|---------|
| tbd   | tbd |      


## Installation Guide
### Run training
```tbd```
### Run Evaluation
tbd
### Use model  
tbd

## Execution:
`todo`



### Setting up Python Environment  
todo remove  

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


## Notes

### Log analysis steps   
- log collection
- log parsing
   Each log message can be parsed into a event template with some specific parameter (variable part) [1].
- feature extraction 
  After parsing logs into separate events we need to further encode them into numerical feature vectors,
  that can be applied on the machine learning algorithms [1]. First we slice the raw logs into a set of log. sequences by using different grouping techniques, including fixed windows, sliding windows , and session windows. Then for each window we generate a feature vector (event count vector), which represents the number of occurance of each event, all feature vector together can form a feature matrix, that is a event count matrix. 
- anomaly detection - Finally the feature matrix can be fed to machine learning models for training, and 
thus henerate model for anomaly detection. 

#### Log parsing 
Logs are plain text that consists of constant parts and variable parts, which may vary among different occurrences. 


## Sources 

[1] Experience report: system log analysis for anomaly detection   
