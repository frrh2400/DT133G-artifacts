import json
import random

from itertools import product

import ollama
import numpy as np
import pandas as pd

from rouge_score import rouge_scorer
from bert_score import score

# LLM model and settings
MODEL = 'phi3:mini'
OPTIONS = {
    'temperature': 0,
    'num_predict': 100
}

# Sampling settings
N_TASKS = 100
SEED = 10

# Load MBPP dataset with embeddings
def load_tasks(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# Count words in reference summary    
def word_count(text):
    return len(text.split())

# Create instruction prompt content
def instruction(query, index):
    n_words = word_count(query['prompt'])

    return (
        'Instruction:\n'
        f'Summarize what the code does in about {n_words} words.'
    )

# Compute cosine similary between embeddings
def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve the most similar problem, excluding the current problem
def top_k(query, index, k=1):
    tasks = []

    for task in index:
        if task['task_id'] != query['task_id']:
            tasks.append(task)

    tasks.sort(
        key=lambda task: similarity(query['embeddings'], task['embeddings']),
        reverse=True
    )

    return tasks[:k]

# Create example prompt content
def example(query, index):
    task, = top_k(query, index)

    code = task['code']
    summary = task['prompt']

    return (
        'Example:\n'
        f'Code -\n{code}\n\n'
        f'Summary -\n{summary}'
    )

# Create context prompt content
def context(query, index):
    tests = '\n'.join(query['test_list'])

    return (
        'Context:\n'
        f'Tests -\n{tests}'
    )

# Add target code that should be summarized
def code(query):
    code = query['code']

    return (
        f'Code:\n{code}'
    )

# Prompt factors used in the factorial experiment
PROMPTS = {
    'instruction': instruction,
    'example': example,
    'context': context
}

# Create prompt settings for instruction, example and context
def prompt_configs():
    configs = []

    for i, flags in enumerate(product([0, 1], repeat=len(PROMPTS)), 1):
        config = {'config': f'C{i}'}

        for name, flag in zip(PROMPTS, flags):
            config[name] = flag

        configs.append(config)

    return configs

# Select prompt content based on the current prompt setting
def prompt_parts(query, config, tasks):
    parts = []

    for name, prompt in PROMPTS.items():
        if config[name]:
            parts.append(prompt(query, tasks))

    return parts

# Build the prompt
def build_prompt(query, config, tasks):
    parts = prompt_parts(query, config, tasks)
    parts.append(code(query))

    return '\n\n'.join(parts)

# Generate a summary using the LLM through Ollama
def generate(prompt):
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options=OPTIONS
    )

    return response

# Compute ROUGE-L scores
def rouge_l(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)['rougeL']

    return {
        'rouge_precision': scores.precision,
        'rouge_recall': scores.recall,
        'rouge_f1': scores.fmeasure
    }

# Generate and evaluate one summary
def evaluate(query, config, index):
    prompt = build_prompt(query, config, index)
    
    response = generate(prompt)
    summary = response.response.strip()
    reference = query['prompt']

    return {
        'task_id': query['task_id'],
        **config,
        'summary': summary,
        'reference': reference,
        'prompt_tokens': response.prompt_eval_count,
        'summary_tokens': response.eval_count,
        **rouge_l(summary, reference)
    }
    
# Compute BERTScore
def bertscore(df):
    P, R, F1 = score(
        df['summary'].tolist(),
        df['reference'].tolist(),
        lang='en'
    )

    df['bertscore_precision'] = P.tolist()
    df['bertscore_recall'] = R.tolist()
    df['bertscore_f1'] = F1.tolist()

    return df

# Save results
def save(df, path):
    df.to_csv(path, index=False)

def main():
    random.seed(SEED)

    # Load problems, sample problems and create prompt settings
    # Update file path
    tasks = load_tasks('mbpp_embeddings.json')
    queries = random.sample(tasks, N_TASKS)
    configs = prompt_configs()

    rows = []

    # Evaluate each sampled problem using each prompt setting
    for query in queries:
        print(query['task_id'])
        for config in configs:
            rows.append(
                evaluate(query, config, tasks)
            )

    # Add BertScore results
    df = pd.DataFrame(rows)
    
    df = bertscore(df)
    
    # Remove generated summaries before saving numerical results
    df = df.drop(columns=['summary', 'reference'])

    # Save results
    # Update file path
    save(df, 'results.csv')

if __name__ == '__main__':
    main()