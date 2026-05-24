# Reproduction package

This package contains the scripts, datasets and analysis files used to reproduce the experiment and statistical analyses.

## Contents

### Python scripts
generate_embeddings.py
- Generates embeddings for MBPP reference summaries using Ollama and embeddinggemma.

generate_results.py
- Uses an LLM to generate summaries with different prompt settings and stores results in `results.csv`.

### R scripts
analysis.R
- Performs repeated-measures ANOVA, estimated marginal means, pairwise comparisons, outlier removal and creates figures.

### Datasets
mbpp.json
- Original MBPP dataset used in the experiment.

mbpp_embeddings.json
- MBPP dataset with generated embeddings used for retrieval.

results.csv
- Evaluation results used in the statistical analyses.

## Requirements

### Python packages
- Python (v.3.14.3)
- ollama (v.0.20.7)
- numpy
- pandas
- rouge_score (v.0.1.2)
- bert_score (v.0.3.13)

### R packages
- R (v.4.6.0)
- afex
- emmeans
- e1071
- ggplot2

### Ollama models
Ollama (v.0.20.7) must be installed locally with the following models:
- embeddinggemma
- phi3:mini

## Workflow
### 1. Generate embeddings
python generate_embeddings.py

### 2. Generate summaries and evaluation results
python generate_results.py

### 3. Run the statistical analysis
analysis.R