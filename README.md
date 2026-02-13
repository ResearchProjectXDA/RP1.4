# Explanation-Driven Self-Adaptations (XDA) Replication Package

## Install Requirements
```pip install -r requirements.txt```

### For macOS and Linux Users
```chmod +x MDP_Dataset_builder/evaluate_adaptations.sh```

## Generate Dataset (Optional)

Inside MDP_Dataset_builder/run.sh and MDP_Dataset_builder/run.bat:
* MAX_SAMPLES: number of samples to generate
* TOTAL_THREADS: number of threads to use for the generation

### For macOS and Linux Users
```
chmod +x MDP_Dataset_builder/run.sh
./run.sh
```

### For Windows Users
```.\run.bat```

## Run Adaptation Tests
Inside main/main.py:
* line 64: you can specify the path to your dataset
* line 83: you can specify the list of requirements to consider
* line 86: you can specify the size of the neighborhood
* line 87: you can specify the number of starting solutions to consider
* line 90: you can specify the target success probabilities for each requirement
* line 157: you can specify the number of tests to do

```python main/main.py```


## Generate Plots
Go to main/resultAnalyzer.py

* line 177: you can specify the path to your results
* lines 179: you can specify the features used
* line 189: you can specify which requirements are used 
```python main/resultAnalyzer.py```

Alternative you can use:
```python main/makeAllPlots.py```
