import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ====================== CONFIG ======================
class Config:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 256
        self.n_embd = 256
        self.n_head = 8
        self.n_layer = 6
        self.dropout = 0.1
        self.learning_rate = 6e-4
        self.max_iters = 10000
        self.eval_interval = 500
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

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
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x

class BabyGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList( )
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
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

# ====================== TRAINING LOOP ======================
def main():
    print(f"Using device: {config.device}")
    
    # TODO: Replace with your actual data
    vocab_size = 100  # placeholder - will change once you have real data
    
    model = BabyGPT(vocab_size, config).to(config.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop placeholder
    for step in range(config.max_iters):
        if step % config.eval_interval == 0:
            print(f"Step {step}: Training...")
    
    print("Training completed!")
    torch.save(model.state_dict(), "babygpt_model.pt")
    print("Model saved as babygpt_model.pt")

if __name__ == "__main__":
    main()