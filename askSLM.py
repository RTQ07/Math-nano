import torch
from transformers import GPT2Tokenizer
from main_model import GPT, GPTConfig
import torch.nn.functional as F

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model checkpoint
checkpoint_path = 'best_model_weights.pt'

# Set up model configuration
config = GPTConfig(vocab_size=70469, block_size=1024, n_layer=16, n_head=18, n_embd=1800, dropout=0.1, bias=True)
model = GPT(config)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate an answer based on a question
def generate_answer(model, tokenizer, device, question, max_length=100, top_p=0.9, temperature=0.7, repetition_penalty=1.5):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
        generated_ids = []
        
        print("Input question:", question)
        print("Input token IDs:", input_ids)
        
        for _ in range(max_length):
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if len(generated_ids) > 0:
                for token_id in set(generated_ids):
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[0, indices_to_remove] = float('-inf')
            
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1).squeeze()
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=1)
            
            print("Generated token ID:", next_token_id.item())
            print("Generated text:", tokenizer.decode(generated_ids))
            print()
        
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text
        
# Prompt the user to ask questions and generate answers
while True:
    question = input("Ask a question (or type 'quit' to exit): ")
    
    if question.lower() == 'quit':
        break
    
    generated_answer = generate_answer(model, tokenizer, device, question, max_length=100, top_p=0.9, temperature=0.7, repetition_penalty=1.5)
    print("Generated Answer:", generated_answer)
    print()