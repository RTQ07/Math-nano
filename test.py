import torch

# Load the tokenized datasets
tokenized_train_dataset = torch.load('tokenized_orca_math_train_dataset.pt')
tokenized_val_dataset = torch.load('tokenized_orca_math_val_dataset.pt')

# Check a few examples from the training dataset
print("Training dataset examples:")
for i in range(5):
    example = tokenized_train_dataset[i]
    print(f"Example {i+1}:")
    print("Input IDs:", example['input_ids'])
    print("Length:", len(example['input_ids']))
    print()

# Check a few examples from the validation dataset
print("Validation dataset examples:")
for i in range(5):
    example = tokenized_val_dataset[i]
    print(f"Example {i+1}:")
    print("Input IDs:", example['input_ids'])
    print("Length:", len(example['input_ids']))
    print()