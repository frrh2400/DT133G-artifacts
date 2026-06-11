import json
import ollama

# Embedding model used for semantic retrieval
MODEL = 'embeddinggemma'

# Load MBPP dataset
def load_tasks(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
    
# Save dataset with embeddings
def save(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

# Generate embeddings
def embed(text):
    response = ollama.embed(
        model=MODEL,
        input=text
    )

    return response.embeddings[0]

def main():
    # Load problems
    tasks = load_tasks('mbpp.json')

    # Generate embeddings for each problem
    for task in tasks:
        task['embeddings'] = embed(task['prompt'])

    # Save results
    # Update file path
    save(tasks, 'mbpp_embeddings.json')

if __name__ == '__main__':
    main()
