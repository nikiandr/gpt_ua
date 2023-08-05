from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# hyperparameters
BATCH_SIZE = 32
SEED = 42
BLOCK_SIZE = 8
EPOCHS = 5000
TRAIN_SUBSET_LENGTH = 1_000_000
TRAIN_PERC = 0.9
EVAL_PERIOD = 500
EVAL_ITERS = 200
EMBED_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----

print(f"Current device: {DEVICE}")

torch.manual_seed(SEED)

DATA_FOLDER_PATH = Path('../data/')
DATA_FILE_PATH = DATA_FOLDER_PATH / "ubertext.wikipedia.filter_rus_gcld+short.text_only.txt"
# DATA_FILE_PATH = DATA_FOLDER_PATH / "input.txt"
with open(DATA_FILE_PATH) as f:
    dataset = f.read()
train_subset = dataset[:TRAIN_SUBSET_LENGTH]
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


train_num = int(TRAIN_PERC * len(train_subset))
train, val = train_subset[:train_num], train_subset[train_num:]

train_tensor = torch.tensor(encode(train), dtype=torch.long)
val_tensor = torch.tensor(encode(val), dtype=torch.long)


def random_batch(split: torch.Tensor, batch_s: int = BATCH_SIZE, bls: int = BLOCK_SIZE) -> tuple[
    torch.Tensor, torch.Tensor]:
    indexes = torch.randint(0, len(split) - bls, (batch_s,))
    x = torch.stack([split[idx:idx + bls] for idx in indexes])
    y = torch.stack([split[idx + 1:idx + bls + 1] for idx in indexes])
    return x, y


class SelfAttentionHead(nn.Module):
    def __init__(self, emb_size: int, head_sz: int, masked: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.head_size = head_sz
        self.masked = masked
        self.key = nn.Linear(self.emb_size, self.head_size, bias=False)  # (B, T, head_size)
        self.query = nn.Linear(self.emb_size, self.head_size, bias=False)  # (B, T, head_size)
        self.value = nn.Linear(self.emb_size, self.head_size, bias=False)  # (B, T, head_size)

    @staticmethod
    def mask(inp: torch.Tensor, device: str) -> torch.Tensor:
        _, t, _ = inp.shape
        tril = torch.tril(torch.ones(t, t).to(device)) == 0
        return inp.masked_fill(tril, float('-inf'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: (B, T, emb_size)
        k, q, v = self.key(x), self.query(x), self.value(x)
        qk = q @ k.transpose(-2, -1)  # (B, T, emb_size) @ (B, emb_size, T)
        if self.masked:
            qk = self.mask(qk, device=DEVICE)
        out = F.softmax(qk, -2) @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_sz: int, num_heads: int = 1, masked: bool = True, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.emb_size = emb_size
        self.head_size = head_sz
        self.num_heads = num_heads
        self.masked = masked
        self.heads = nn.ModuleList([SelfAttentionHead(self.emb_size, self.head_size) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], -1)


class AttentionBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 1, masked: bool = True, *args, **kwarg):
        super().__init__(*args, **kwarg)
        if emb_size % num_heads != 0:
            raise ValueError(f"Embedding size ({emb_size}) is not divisible by number of heads ({num_heads})")
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.masked = masked
        self.attention_heads = MultiHeadAttention(self.emb_size, self.emb_size // self.num_heads, self.num_heads)
        self.fc = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.attention_heads(x)
        out = out + self.fc(out)
        return out


class BiGramModel(nn.Module):
    def __init__(self, dict_size: int = None, embedding_size: int = None, blck_size: int = None, num_heads: int = 5):
        super().__init__()
        self.num_heads = num_heads
        if dict_size is None:
            self.dict_size = len(vocabulary)
        else:
            self.dict_size = dict_size
        if embedding_size is None:
            self.embedding_size = EMBED_SIZE
        else:
            self.embedding_size = embedding_size
        if blck_size is None:
            self.block_size = BLOCK_SIZE
        else:
            self.block_size = blck_size
        self.embeddings = nn.Embedding(self.dict_size, self.embedding_size)
        self.positional_embeddings = nn.Embedding(self.block_size, self.embedding_size)
        self.att = nn.Sequential(
            AttentionBlock(self.embedding_size, self.embedding_size),
            AttentionBlock(self.embedding_size, self.embedding_size),
            AttentionBlock(self.embedding_size, self.embedding_size),
        )
        self.lm_head = nn.Linear(self.embedding_size, self.dict_size)

    def forward(self, idxs: torch.Tensor, targs: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idxs.shape
        # Input: (B, T)
        # Embeds: (B, T, embedding_size)
        embeds = self.embeddings(idxs)
        # Positional embeds: (T, embedding_size)
        positional_embeds = self.positional_embeddings(torch.arange(t, device=DEVICE))
        # Embeds + positional embeds: (B, T, embedding_size * head_size), broadcasting on dimension B
        # Logits: (B, T, C)
        embeds = self.att(embeds + positional_embeds)
        lgits = self.lm_head(embeds)
        if targs is None:
            loss = None
        else:
            B, T, C = lgits.shape
            lgits = lgits.view(B * T, C)
            targs = targs.view(B * T)
            loss = F.cross_entropy(lgits, targs)
        return lgits, loss

    def generate(self, idxs: torch.Tensor, max_token_gen: int = 20):
        # idx of size (B, T)
        for _ in tqdm(range(max_token_gen)):
            # crop idx to be only last block
            idx_cond = idxs[:, -self.block_size:]
            # logits of size (B, T, C)
            logits, _ = self(idx_cond)
            # logit of size (B, C)
            logit = logits[:, -1, :]
            # probs of size (B, C)
            probs = F.softmax(logit, dim=-1)
            # next_characters of size (B, 1)
            next_characters = torch.multinomial(probs, num_samples=1)
            idxs = torch.cat([idxs, next_characters], 1)
        return idxs


model = BiGramModel().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in [("train", train_tensor),
                  ("val", val_tensor)]:
        avg_loss = 0
        for iter in range(EVAL_ITERS):
            xb, yb = random_batch(split[1], batch_s=BATCH_SIZE, bls=BLOCK_SIZE)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, loss = model(xb, yb)
            avg_loss += loss
        out[split[0]] = avg_loss / EVAL_ITERS
    model.train()
    return out


for epoch in tqdm(range(EPOCHS)):
    if epoch % EVAL_PERIOD == 0:
        losses = estimate_loss()
        print(f"Iteration: {epoch}, validation loss: {losses['val']}, train loss: {losses['train']}")
    xb, yb = random_batch(train_tensor, batch_s=BATCH_SIZE, bls=BLOCK_SIZE)
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
print(f"Final loss: {loss.item()}")

idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
idx_gen = model.generate(idx, max_token_gen=10000)
print(decode(idx_gen[0].tolist()))
