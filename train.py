import os
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from src.Transformer import Transformer

load_dotenv()

# Detect and use the best available hardware (CUDA, MPS, or CPU)
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Final Hyperparameters (based on Andrej Karpathy's video)
block_size = 256
batch_size = 64
learning_rate = 3e-4
train_iterations = 5000
eval_iterations = 500
temperature = 1.0
number_embeddings = 384
num_heads = 6
dropout_rate = 0.2

shakespeare_path = os.getenv("SHAKESPEARE_PATH")

with open(shakespeare_path, "r") as file:
    text = file.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

def get_batch(data):
    # Generate random starting points
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack the chunks and move to the target device
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)

def main():
    # Initialize and move model to device
    model = Transformer(
        chars=chars, 
        temperature=temperature, 
        num_embeddings=number_embeddings, 
        block_size=block_size, 
        num_heads=num_heads, 
        dropout_rate=dropout_rate
    )
    model.to(device)

    # Encode data and split into train/val
    data = torch.tensor(model.encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    # Move datasets to device
    train_data = data[:n].to(device)
    val_data = data[n:].to(device)

    # Use AdamW (standard for Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training model...")
    model.train() # Turn on dropout
    
    for step in range(train_iterations):
        xb, yb = get_batch(train_data)
        
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = F.cross_entropy(logits, targets)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}: training loss {loss.item():.4f}")
            print("\nSaving model...")
            torch.save(model.state_dict(), 'transformer_model.pth')


    print("\nGeneration Started:")
    # generate() now correctly finds the model's device and sets eval() mode
    model.generate()

if __name__ == "__main__":
    main()
