from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch

# Load the dataset
dataset = load_dataset("microsoft/orca-math-word-problems-200k")

# Split the dataset into training and validation sets
train_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset, val_dataset = train_dataset['train'], train_dataset['test']

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Define the maximum sequence length
max_length = 1024

# Tokenize the dataset
def tokenize_function(example):
    question = example['question']
    answer = example['answer']

    # Concatenate question and answer with a separator token
    input_text = f"{question} {tokenizer.eos_token} {answer}"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)

    return {'input_ids': input_ids}

tokenized_train_dataset = train_dataset.map(tokenize_function, num_proc=4, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(tokenize_function, num_proc=4, remove_columns=val_dataset.column_names)

# Save the tokenized datasets to .pt files
torch.save(tokenized_train_dataset, 'tokenized_orca_math_train_dataset.pt')
torch.save(tokenized_val_dataset, 'tokenized_orca_math_val_dataset.pt')