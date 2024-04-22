# Is Mamba Capable of In-Context Learning?
## Section 4: Investigation of Simple NLP Tasks
In this section we investigated the ability of Mamba to perform in-context learning on simple NLP tasks. 
We used the tasks proposed by Hendel et al. (2023) (see task_vector_README.md for more details).

## Environment Setup
We modified the environment file [environment.yml](environment.yml) provided by Hendel et al. (2023) to create a python environment with all the required dependencies. To create the environment, run the following command:
```
conda env create -f environment.yml -n simple_nlp_tasks
```
Then, activate the environment:
```
conda activate simple_nlp_tasks
```

## Experiments
The following experiments reproduce the Mamba and RKWV results in Figure 6 and 7.
We use the experimental results from Hendel et al. (2023).

### Running the experiments
To run the experiments for Figure 6 and 7, run the following command:
```
$ chmod +x context_length_eval.sh
$ ./context_length_eval.sh
```

The results will be saved to `outputs/results/main/context_length`.
Please move the results with `*_num_ex_5.pkl` to `outputs/results/main/camera_ready`.


### Plotting the figures
To generate the figures from the paper, run the following command:
```
python scripts/figures/main.py
```
The figures will be saved to `outputs/figures`.