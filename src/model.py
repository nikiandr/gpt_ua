import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def random_batch(split: torch.Tensor, batch_s: int,
                 bls: int) -> tuple[torch.Tensor, torch.Tensor]:
    indexes = torch.randint(0, len(split) - bls, (batch_s,))
    x = torch.stack([split[index:index + bls] for index in indexes])
    y = torch.stack([split[index + 1:index + bls + 1] for index in indexes])
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
    def __init__(self, emb_size: int, num_heads: int = 1, masked: bool = True, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = self.emb_size // num_heads
        self.masked = masked
        self.heads = nn.ModuleList([SelfAttentionHead(self.emb_size, self.head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(self.head_size * self.num_heads, self.head_size * self.num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], -1)
        out = self.proj(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads, masked: bool = True, *args, **kwarg):
        super().__init__(*args, **kwarg)
        if emb_size % num_heads != 0:
            raise ValueError(f"Embedding size ({emb_size}) is not divisible by number of heads ({num_heads})")
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.masked = masked
        self.attention_heads = MultiHeadAttention(self.emb_size, self.num_heads)
        self.fc = nn.Sequential(
            nn.Linear(self.emb_size, 4 * self.emb_size),
            nn.ReLU(),
            nn.Linear(4 * self.emb_size, self.emb_size)
        )
        self.ln1 = nn.LayerNorm(self.emb_size)
        self.ln2 = nn.LayerNorm(self.emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.attention_heads(self.ln1(x))
        out = out + self.fc(self.ln2(out))
        return out


class GPT(nn.Module):
    def __init__(self, dict_size: int, embedding_size: int, block_size: int, num_heads: int, block_number: int):
        super().__init__()
        self.num_heads = num_heads
        self.dict_size = dict_size
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.block_number = block_number
        self.embeddings = nn.Embedding(self.dict_size, self.embedding_size)
        self.positional_embeddings = nn.Embedding(self.block_size, self.embedding_size)
        self.att = nn.Sequential(
            *[AttentionBlock(self.embedding_size, self.embedding_size) for _ in range(self.block_number)]
        )
        self.lm_head = nn.Linear(self.embedding_size, self.dict_size)

    def forward(self, idxs: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idxs.shape
        # Input: (B, T)
        # Embeds: (B, T, embedding_size)
        embeds = self.embeddings(idxs)
        # Positional embeds: (T, embedding_size)
        positional_embeds = self.positional_embeddings(torch.arange(t, device=DEVICE))
        # Embeds + positional embeds: (B, T, embedding_size * head_size), broadcasting on dimension B
        # Logits: (B, T, C)
        embeds = self.att(embeds + positional_embeds)
        gits = self.lm_head(embeds)
        if targets is None:
            cur_loss = None
        else:
            b, t, c = gits.shape
            gits = gits.view(b * t, c)
            targets = targets.view(b * t)
            cur_loss = F.cross_entropy(gits, targets)
        return gits, cur_loss

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_token_gen: int = 20):
        # idx of size (B, T)
        for _ in tqdm(range(max_token_gen)):
            # crop idx to be only last block
            idx_cond = ids[:, -self.block_size:]
            # logits of size (B, T, C)
            logs, _ = self(idx_cond)
            # logit of size (B, C)
            logit = logs[:, -1, :]
            # probs of size (B, C)
            probs = F.softmax(logit, dim=-1)
            # next_characters of size (B, 1)
            next_characters = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_characters], 1)
        return ids

    @torch.no_grad()
    def estimate_loss(self, train_set: torch.Tensor, val_set: torch.Tensor, batch_size: int, eval_iters: int):
        out = {}
        self.eval()
        for split in [("train", train_set),
                      ("val", val_set)]:
            avg_loss = 0
            for _ in range(eval_iters):
                x_batch, y_batch = random_batch(split[1], batch_s=batch_size, bls=self.block_size)
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                _, cur_loss = self(x_batch, y_batch)
                avg_loss += cur_loss
            out[split[0]] = avg_loss / eval_iters
        self.train()
        return out
