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


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
PAD_TOKEN_ID = tokenizer.eos_token_id 
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:640'

train_losses = []
val_losses = []


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up AWS S3 client
s3 = boto3.client('s3')

# Specify the S3 bucket and file paths
bucket_name = 'math-nano'
train_file_path = 'mathpile/train'
val_file_path = 'mathpile/val'

# Download the tokenized data from S3
def download_from_s3(file_path):
    response = s3.get_object(Bucket=bucket_name, Key=file_path)
    data = response['Body'].read()
    return torch.load(io.BytesIO(data))

# Load the tokenized data
train_data = download_from_s3(train_file_path)
val_data = download_from_s3(val_file_path)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up model configuration
config = GPTConfig(vocab_size=70469, block_size= 1024, n_layer=16, n_head=18, n_embd=1800, dropout=0.1, bias=True)
model = GPT(config)
model.to(device)

# Set up optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Set up loss function
criterion = nn.CrossEntropyLoss()

# Set up training parameters
num_epochs = 40
#save_interval = 2  # Save weights every 2 epochs
log_interval = 1000  # Log information every 100 batches
batch_size = 1


# Set up mixed precision
scaler = amp.GradScaler()

best_val_loss = float('inf')

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

# Set the model to train mode
model.train()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    num_examples = 0
    
    for batch_idx in range(0, len(train_data['input_ids']), batch_size):
        input_ids = train_data['input_ids'][batch_idx:batch_idx+batch_size].to(device)
        attention_mask = (input_ids != PAD_TOKEN_ID).float().to(device)
        targets = input_ids.clone()  # Create a copy of input_ids
        targets[:, :-1] = input_ids[:, 1:]  # Shift the tokens by one position to the left
        targets[:, -1] = config.vocab_size  # Set the last token to -100 (ignore index)


        optimizer.zero_grad()
        
        # Mixed precision
        with amp.autocast():
            logits, _ = model(input_ids, attention_mask=attention_mask, targets=targets, config=config)  # Pass the config object
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=config.vocab_size)
            
        # Scale loss and backward
        scaler.scale(loss).backward()

        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        num_examples += input_ids.size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            avg_train_loss = train_loss / num_examples
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_data['input_ids'])}], Train Loss: {avg_train_loss:.4f}")
            wandb.log({"train_loss": avg_train_loss})
            train_loss = 0.0
            num_examples = 0
  

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_idx in range(0, len(val_data['input_ids']), batch_size):
            input_ids = val_data['input_ids'][batch_idx:batch_idx+batch_size].to(device)
            attention_mask = (input_ids != PAD_TOKEN_ID).float().to(device)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = config.vocab_size  # Set the last token to -100 (ignore index)
            
            logits, _ = model(input_ids, attention_mask=attention_mask, targets=targets, config=config)   # Pass the config object
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=config.vocab_size)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / ((len(val_data['input_ids']) - 1) // batch_size + 1)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    wandb.log({"val_loss": avg_val_loss})

    # Save the best model weights
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        torch.save(best_model_weights, 'best_model_weights.pt')
        #wandb.save('best_model_weights.pt')
        logger.info(f"Saved best model weights with validation loss: {best_val_loss:.4f}")
    
    
    # Scheduler Step
    scheduler.step()


logger.info("Training completed.")