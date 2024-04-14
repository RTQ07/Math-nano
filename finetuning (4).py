import wandb
import io
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from main_model import GPT, GPTConfig
from torch.nn import functional as F
import boto3
import logging
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
import torch.cuda.amp as amp
from transformers import GPT2Tokenizer
import cProfile
from torch.utils.data import Dataset

class MathDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        return input_ids

# Set the maximum split size to 1024 MB (1 GB)
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

os.environ["NUMEXPR_MAX_THREADS"] = "30"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up model configuration
config = GPTConfig(vocab_size=70469, block_size=1024, n_layer=16, n_head=18, n_embd=1800, dropout=0.1, bias=True)

# Set up training parameters
num_epochs = 70
log_interval = 1000
batch_size = 1
#accumulation_steps = 8

# Load the tokenized data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
PAD_TOKEN_ID = tokenizer.eos_token_id 
train_data = torch.load('tokenized_orca_math_train_dataset.pt')
val_data = torch.load('tokenized_orca_math_val_dataset.pt')

# Create dataset instances
train_dataset = MathDataset(train_data)
val_dataset = MathDataset(val_data)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load the saved model weights and other metrics
checkpoint = torch.load('best_model_w.pt')
model = GPT(config)
model.load_state_dict(checkpoint)
model.to(device)

# Set up optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Set up loss function
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')

#best_val_loss = checkpoint['loss']
#start_epoch = checkpoint['epoch'] + 1


# Set up mixed precision
scaler = amp.GradScaler()

# Initialize wandb
wandb.init(project="math-nanogpt", entity="mathrtq")


# Log model architecture and hyperparameters
wandb.config.update({
    "model_architecture": model.__class__.__name__,
    "num_layers": config.n_layer,
    "num_heads": config.n_head,
    "embedding_size": config.n_embd,
    "vocab_size": config.vocab_size,
    "batch_size": batch_size,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "num_epochs": num_epochs,
})

# def log_gradients_and_activations(model, logger):
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             param.register_hook(lambda grad, name=name: logger.info(f"Gradient {name}: {grad.abs().mean()}"))

#     for name, module in model.named_modules():
#         if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
#             module.register_forward_hook(lambda module, input, output, name=name: logger.info(f"Activation {name}: {output.abs().mean()}"))

# # Call the logging function before the training loop
# log_gradients_and_activations(model, logger)



# # Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    num_examples = 0

    for batch_idx, input_ids in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = (input_ids != PAD_TOKEN_ID).float().to(device)
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]
        targets[:, -1] = config.vocab_size

        optimizer.zero_grad()

        with amp.autocast():
            logits, _ = model(input_ids, attention_mask=attention_mask, targets=targets, config=config)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=config.vocab_size)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
            
        train_loss += loss.item()
        num_examples += input_ids.size(0)

        if (batch_idx + 1) % log_interval == 0:
            avg_train_loss = train_loss / num_examples
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Train Loss: {avg_train_loss:.4f}")
            wandb.log({"train_loss": avg_train_loss})
            train_loss = 0.0
            num_examples = 0

    #Evaluate on validation set and save the best model weights
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
         for batch_idx, input_ids in enumerate(val_loader):
            input_ids = input_ids.to(device)
            attention_mask = (input_ids != PAD_TOKEN_ID).float().to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = config.vocab_size  # Set the last token to -100 (ignore index)
            
            logits, _ = model(input_ids, attention_mask=attention_mask, targets=targets, config=config)   # Pass the config object
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=config.vocab_size)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    wandb.log({"val_loss": avg_val_loss})

    # Save the best model weights
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        #torch.save({
            #'epoch': epoch,
            #'model_state_dict': best_model_weights,
            #'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            #'loss': avg_val_loss,
            #}, 'model_info.pt')
        torch.save(best_model_weights, 'best_model_weights.pt')
        #wandb.save('best_model_weights.pt')
        logger.info(f"Saved best model weights with validation loss: {best_val_loss:.4f}")
    

    # Scheduler step
    scheduler.step()


logger.info("Fine-tuning completed.")