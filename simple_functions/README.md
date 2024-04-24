# Is Mamba Capable of In-Context Learning?
## Section 3: Investigation of Simple Function Classes

## Environment Setup
We modified the requirements file [requirements.txt](requirements.txt) provided by Garg et al. (2022) to create a python environment with all the required dependencies. To create the environment, run the following command:
```
conda create -n simple_functions python=3.8
conda activate simple_functions
pip install -r requirements.txt
``` 

## Experiments
To run the experiments please run
```
cd src
chmod +x experiments.sh
./experiments.sh
```
This will run the experiments sequentially. For parallel execution on a cluster, please make the necessary changes.
The cost of each run on an Nvidia RTX 2080 is about 1 day.
The results will be saved to `outputs/models/`.

## Evaluating the models
To evaluate the models, run the following command:
```
cd src
python eval.py 200
```
The results will be saved as metrics.json in every run directory.

## Plotting the figures
To generate the figures from the paper, run the following command
For plotting please follow the instructions in eval.ipynb

