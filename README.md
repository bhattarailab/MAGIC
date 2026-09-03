# MAGIC
## Setup Instruction
This single branch is used for all the points in pipeline so, a separate virtual environment is required to handle conflicting requirement of different steps.

To manage this, the project has separate virtual environment configs for various steps mentioned inside `/envs`. This can be easily used with the help of `uv`.

```python
# How to run a script
uv run --project ./envs/<enviornment-folder> python script.py --arg value
```

All the projects are;
1. `default` used for most of scripts which dependency is pretty linient.
2. `llava-med` used for running the attention extraction for LLavaMed models
3. `green` used for running the GREEN model

## Pipeline
*For any stage of pipeline use --help to know about the arguments that the script accepts*

The pipeline is as follows;
1. `vlm-inference.py` used to extraction attention and embeddingss across heads and layers given a dataset and a model. 
2. `green-eval.py` used to genrate the ground truth for hallucination detection. For each sample, given a question, generated answer and reference answer, wherein question and refernce answer are provided by dataset, one can use GREEN model to discover if there are any clinically signficant errors between the refernce or ground truth. 
3. `inference-gini.py` used to compute the GINI values either by masking or not masking the attention sinks. This also trains a detector for hallucination detection and provides the evaluation metrics.
