#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the maximum sequence length and truncation flag
max_length = 1024
truncation = True

# Tokenization function
def tokenize_chunk(chunk):
    return tokenizer(chunk, max_length=max_length, truncation=truncation, return_tensors='pt')

# Checkpoint saving function
def save_checkpoint(encoded_chunks, checkpoint_path):
    torch.save(encoded_chunks, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Tokenize the training dataset in chunks
chunk_size = 1024 * 1024  # 1 MB chunks
encoded_chunks = []
checkpoint_interval = 10000  # Save checkpoint every 10000 chunks
checkpoint_path = 'mathpile_train_encoded_checkpoint.pt'

# Check if a checkpoint exists
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    encoded_chunks = torch.load(checkpoint_path)
    print(f"Loaded {len(encoded_chunks)} chunks from checkpoint")

with open('temp_preprocessed_dataset.txt', 'r') as file:
    print("Tokenizing the dataset...")
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        encoded_chunk = tokenize_chunk(chunk)
        encoded_chunks.append(encoded_chunk)
        print(f"Processed {len(encoded_chunks)} chunks")
        
        # Save checkpoint periodically
        if len(encoded_chunks) % checkpoint_interval == 0:
            save_checkpoint(encoded_chunks, checkpoint_path)

# Save the final checkpoint
save_checkpoint(encoded_chunks, checkpoint_path)

# Concatenate the encoded chunks
encoded_dataset = {}
for key in encoded_chunks[0].keys():
    encoded_dataset[key] = torch.cat([chunk[key] for chunk in encoded_chunks], dim=0)

# Save the encoded dataset
print("Saving the encoded dataset...")
torch.save(encoded_dataset, 'mathpile_train_encoded.pt')
print("Encoded dataset saved.")

