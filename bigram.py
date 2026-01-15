import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions? #REF:16:00
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
logits_preview = True
logits_preview_interval = 300
heatmap_preview = True
heatmap_preview_interval = 600
heatmap_topk = 20
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers i.e. tokenizer --> Google uses BPE tokenizer, sub-word encodings (shorter encoded sequences)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits; converting all text to a tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices for the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # fetch block_size tokens for each starting index
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by one, predict next char
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # no gradient tracking needed, saves memory and computations
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # doesn't do anything for bigram, but good practice. In NN, this would re-enable dropout/batchnorm
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # input: vocab_size, output: vocab_size (logits for each token in vocab)

    # forward pass to get logits and loss
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C), Batch, Time, Channel (vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshape to (B*T, C) for cross-entropy in pytorch
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # computes softmax internally

        return logits, loss

    # generate method to sample from the model, and produce new text
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step, i.e. predict the next token
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), in each batch, sample one token
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device) # move model to GPU if available 

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits_full = model.token_embedding_table(xb)
    logits, loss = model(xb, yb)
    if logits_preview and iter % logits_preview_interval == 0:
        # quick peek at logits distribution for the final time step of a single sample
        row = logits_full[0, -1].detach().float().cpu()
        probs = F.softmax(row, dim=-1)
        print(
            f"logits preview @ step {iter}: "
            f"min={row.min():.3f} max={row.max():.3f} mean={row.mean():.3f} std={row.std():.3f}"
        )
        # token:prob pairs (vocab order) for a single prediction row
        pairs = " ".join([f"{itos[i]}:{probs[i]:.3f}" for i in range(vocab_size)])
        print(f"probs row @ step {iter}: {pairs}")
    if heatmap_preview and iter % heatmap_preview_interval == 0:
        # heatmap of top-k probs for the final time step of the first sample
        row = logits_full[0, -1].detach().float().cpu()
        probs = F.softmax(row, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=min(heatmap_topk, vocab_size))
        top_probs = top_probs.numpy().reshape(1, -1)
        plt.figure(figsize=(10, 1.8))
        plt.imshow(top_probs, aspect='auto', cmap='viridis', vmin=0.0, vmax=top_probs.max())
        plt.yticks([])
        plt.xticks(
            range(top_probs.shape[1]),
            [itos[i] for i in top_idx.tolist()],
            fontsize=8
        )
        plt.colorbar(label='prob')
        plt.tight_layout()
        plt.savefig(f'logits_heatmap_step_{iter}.png', dpi=150)
        plt.close()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model, conditioned on a starting context, which is just a single zero token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generate 500 tokens, thus the output will be of shape (1, 501) and look like [[token0, token1, token2, ...]]
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# bigram model; only looks at the previous token to predict the next token
# in the transformer, we will look at the entire context of previous tokens to predict the next token
