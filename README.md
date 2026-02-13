# Explanation-Driven Self-Adaptations Replication Package

## Repository Structure

This repository is divided into the following directories:

### 📁 `main`
Contains the core implementation of the project, including:
- Explainer algorithms
- Main execution scripts
- Utility/helper modules
- Additional supporting code

### 📁 `datasets`
Contains all datasets used for training, testing, and evaluation.

### 📁 `results`
Stores experiment outputs in `.csv` format, organized into subfolders.

### 📁 `MDP_dataset_builder`
Provides the necessary tools and scripts to generate custom personalized datasets.

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

## Run Experiments
Inside main/main.py:
* line 64: you can specify the path to your dataset  
  defult: ```ds = pd.read_csv('../datasets/new_dataset.csv')```
* line 83: you can specify the list of requirements to consider  
  default: ```reqs = ["req_0", "req_1", "req_2", "req_3"]```
* line 86: you can specify the size of the neighborhood  
  default: ```n_neighbors = 10```
* line 87: you can specify the number of starting solutions to consider  
  default: ```n_startingSolutions = 10```
* line 90: you can specify the target success probabilities for each requirement  
  default: ```targetConfidence = np.full((1, n_reqs), 0.8)[0]```
* line 157: you can specify the number of tests to do  
  default: ```testNum = 20```
* line 335: you can specify the path to the results  
  default: ```path = "../results"```

```python main/main.py```

The results will be saved in .csv format in the path specified in line 335. There will be saved the adaptations, the relative confidences, scores, times and other useful metrics.

## Generate Plots
Go to main/resultAnalyzer.py

* line 177: you can specify the path to your results  
  default: ```pathToResults = "../results/" #sys.argv[1]```
* lines 179: you can specify the features used  
  default: ``` featureNames = ["cruise speed","image resolution","illuminance","controls responsiveness","power","smoke intensity","obstacle size","obstacle distance","firm obstacle"]```
* line 189: you can specify which requirements are used  
  default: ```reqs = ["req_0", "req_1", "req_2", "req_3"]```
  
```python main/resultAnalyzer.py```  
The plots will be saved as .png images in the folder results, as specified in line 177.
Alternatively, you can use this [notebook](main/notebooks/ResultsNB.ipynb) by specifying the same information.
