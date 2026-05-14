import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ====================== CONFIG ======================
class Config:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 256          # context length
        self.n_embd = 320              # embedding size
        self.n_head = 8
        self.n_layer = 8
        self.dropout = 0.1
        self.learning_rate = 6e-4
        self.max_iters = 8000
        self.eval_interval = 400
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

print(f"Training on: {config.device}")

# ====================== DATA LOADER ======================
class CharDataset:
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x.to(config.device), y.to(config.device)

# ====================== MODEL ======================
class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class BabyGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.pos_embedding(torch.arange(T, device=idx.device))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ====================== MAIN TRAINING ======================
def main():
    # Load data
    with open('training.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    dataset = CharDataset(text, config.block_size)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Total tokens: {len(dataset.data):,}")

    model = BabyGPT(dataset.vocab_size, config).to(config.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.max_iters):
        if step % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = dataset.get_batch(8)
                _, loss = model(xb, yb)
                print(f"Step {step:4d} | val loss {loss.item():.4f}")
            model.train()

        xb, yb = dataset.get_batch(config.batch_size)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab': (dataset.stoi, dataset.itos)
    }, "babygpt_model.pt")
    
    print("\nTraining finished! Model saved as babygpt_model.pt")

if __name__ == "__main__":
    main()