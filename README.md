# lisp_benchmarking
Benchmarking LLMs For Scheme on DeepMind's CodeContests Dataset

## About
We aim to test LLMs abilities on programming tasks with languages like Scheme. We make use of the codecontests dataset from deepmind.

## Test
Dataset can be downloaded from huggingface, and suitably converted to a csv using the [data/extract](data/extract.py) file.
Configure models, input file and logging files as needed in [evaluation](eval.py) file.

## Scheme Function Levels
To enquire into the reasoning abilities of LLMs on these tasks and owing to the recursive nature of Scheme, we propose using a sorting of function into different levels as a rough measure of problem difficulty.
Output Scheme code can be passed to [parse_scheme](parse_scheme.py), which looks into the function dependencies in the program and sorts them into levels.
Usage : python parse_scheme.py -- scheme_file
