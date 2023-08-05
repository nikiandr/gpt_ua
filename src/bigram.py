from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
seed = 42
block_size = 16
epochs = 10000
train_subset_length = 1_000_000
train_perc = 0.9
eval_period = 200
eval_iters = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

DATA_FOLDER_PATH = Path('../data/')
DATA_FILE_PATH = DATA_FOLDER_PATH / "ubertext.wikipedia.filter_rus_gcld+short.text_only.txt"
with open(DATA_FILE_PATH) as f:
    dataset = f.read()
train_subset = dataset[:train_subset_length]
vocabulary = sorted(list(set(train_subset)))

stoi = {s: i for i, s in enumerate(vocabulary)}
itos = {i: s for i, s in enumerate(vocabulary)}


def encode(text: str, char_map=None) -> list[int]:
    if char_map is None:
        char_map = stoi
    return [char_map[char] for char in text]


def decode(embedding: list[int], index_map=None) -> str:
    if index_map is None:
        index_map = itos
    return ''.join([index_map[idx] for idx in embedding])


train_num = int(train_perc * len(train_subset))
train, val = train_subset[:train_num], train_subset[train_num:]

train_tensor = torch.tensor(encode(train), dtype=torch.long)
val_tensor = torch.tensor(encode(val), dtype=torch.long)


def random_batch(split: torch.Tensor, batch_s: int = batch_size, bls: int = block_size) -> tuple[torch.Tensor, torch.Tensor]:
    indexes = torch.randint(0, len(split) - bls, (batch_s,))
    x = torch.stack([split[idx:idx + bls] for idx in indexes])
    y = torch.stack([split[idx + 1:idx + bls + 1] for idx in indexes])
    return x, y


class BiGramModel(nn.Module):
    def __init__(self, dict_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(dict_size, dict_size)

    def forward(self, idxs: torch.Tensor, targs: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        lgits = self.embeddings(idxs)
        if targs is None:
            loss = None
        else:
            B, T, C = lgits.shape
            lgits = lgits.view(B * T, C)
            targs = targs.view(B * T)
            loss = F.cross_entropy(lgits, targs)
        return lgits, loss

    def generate(self, idx: torch.Tensor, max_token_gen: int = 20):
        # idx of size (B, T)
        for _ in range(max_token_gen):
            # logits of size (B, T, C)
            logits, _ = self(idx)
            # logit of size (B, C)
            logit = logits[:, -1, :]
            # probs of size (B, C)
            probs = F.softmax(logit, dim=-1)
            # next_characters of size (B, 1)
            next_characters = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_characters], 1)
        return idx


model = BiGramModel(len(vocabulary)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in [("train", train_tensor),
                  ("val", val_tensor)]:
        avg_loss = 0
        for iter in range(eval_iters):
            xb, yb = random_batch(split[1], batch_s=batch_size, bls=block_size)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            avg_loss += loss
        out[split[0]] = avg_loss / eval_iters
    model.train()
    return out


for epoch in range(epochs):
    if epoch % eval_period == 0:
        losses = estimate_loss()
        print(f"Iteration: {epoch}, validation loss: {losses['val']}, train loss: {losses['train']}")
    xb, yb = random_batch(train_tensor, batch_s=batch_size, bls=block_size)
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
print(f"Final loss: {loss.item()}")

idx = torch.zeros((1, 1), dtype=torch.long).to(device)
idx_gen = model.generate(idx, max_token_gen=1000)
print(decode(idx_gen[0].tolist()))
