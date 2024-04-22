# Is Mamba Capable of In-Context Learning?
This is the official code repository for our paper.

## Abstract:
> This work provides empirical evidence that Mamba, a newly proposed selective structured state space model, has similar in-context learning (ICL) capabilities as transformers. We evaluated Mamba on tasks involving simple function approximation as well as more complex natural language processing problems. Our results demonstrate that across both categories of tasks, Mamba matches the performance of transformer models for ICL. Further analysis reveals that like transformers, Mamba appears to solve ICL problems by incrementally optimizing its internal representations. Overall, our work suggests that Mamba can be an efficient alternative to transformers for ICL tasks involving longer input sequences.

## Experimental Setup

Our experiments are split into two parts - simple function approximation and natural language processing (NLP) tasks. 

### Section 3: Simple Function Approximation
The code from the experiments is based on the code by Garg et al. (2022)
Please see the detailed readme under /simple_functions/README.md for details on how to reproduce our experiments

### Section 4: Investigation of Simple NLP Tasks
The code from the experiments is based on the code by Hendel et al. (2023)
Please see the detailed readme under /simple_nlp_tasks/README.md for details on how to reproduce our experiments

## References:

> Garg, Shivam, et al. "What can transformers learn in-context? a case study of simple function classes." Advances in Neural Information Processing Systems 35 (2022): 30583-30598.

> Hendel, Roee, Mor Geva, and Amir Globerson. "In-Context Learning Creates Task Vectors." Findings of the Association for Computational Linguistics: EMNLP 2023. 2023.