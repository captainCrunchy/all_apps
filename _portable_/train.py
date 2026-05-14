import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ====================== CONFIG ======================
class Config:
    def __init__(self):
        self.batch_size = 64
        self.block_size = 256
        self.n_embd = 384
        self.n_head = 8
        self.n_layer = 8
        self.dropout = 0.15          # increased
        self.learning_rate = 4e-4
        self.max_iters = 10000
        self.eval_interval = 500
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# ====================== DATA ======================
class CharDataset:
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # Split 90/10
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split, batch_size):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
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

# ====================== GENERATION ======================
@torch.no_grad()
def generate(model, dataset, prompt="", max_new_tokens=500):
    model.eval()
    idx = torch.tensor([dataset.stoi[c] for c in prompt], dtype=torch.long, device=config.device).unsqueeze(0)
    if len(idx[0]) == 0:
        idx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return ''.join([dataset.itos[i] for i in idx[0].tolist()])

# ====================== MAIN ======================
def main():
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
                xb, yb = dataset.get_batch('val', batch_size=16)
                _, val_loss = model(xb, yb)
                print(f"Step {step:5d} | val loss {val_loss.item():.4f}")
            model.train()

        xb, yb = dataset.get_batch('train', config.batch_size)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    # Final generation test
    print("\n" + "="*60)
    print("GENERATED TEXT:")
    print("="*60)
    print(generate(model, dataset, prompt="The sea ", max_new_tokens=800))

if __name__ == "__main__":
    main()