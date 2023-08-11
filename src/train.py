import torch
from tqdm.auto import tqdm
import bigram_char_model as bg
import os

from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Current folder: {os.getcwd()}")
print(f"Current device: {DEVICE}")

torch.manual_seed(SEED)

# DATA_FILE_PATH = DATA_FOLDER_PATH / "input.txt"
with open(DATA_FILE_PATH) as f:
    train_subset = f.read()
train_subset = train_subset[:TRAIN_SUBSET_LENGTH]
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
    return ''.join([index_map[cur_id] for cur_id in embedding])


train_num = int(TRAIN_PERC * len(train_subset))
train, val = train_subset[:train_num], train_subset[train_num:]

train_tensor = torch.tensor(encode(train), dtype=torch.long)
val_tensor = torch.tensor(encode(val), dtype=torch.long)

# ex-model place

model = bg.BiGramModel(dict_size=len(vocabulary),
                       embedding_size=EMBED_SIZE,
                       block_size=BLOCK_SIZE,
                       num_heads=NUM_HEADS,
                       block_number=BLOCK_NUMBER).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in tqdm(range(EPOCHS)):
    if epoch % EVAL_PERIOD == 0:
        losses = model.estimate_loss(train_set=train_tensor,
                                     val_set=val_tensor,
                                     batch_size=BATCH_SIZE,
                                     eval_iters=EVAL_ITERS)
        print(f"Iteration: {epoch}, validation loss: {losses['val']}, train loss: {losses['train']}")
    xb, yb = bg.random_batch(train_tensor, batch_s=BATCH_SIZE, bls=BLOCK_SIZE)
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
print(f"Final loss: {loss.item()}")
torch.save(model.state_dict(), MODEL_PATH)

print("Generating example: ")
idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
idx_gen = model.generate(idx, max_token_gen=1000)
print(decode(idx_gen[0].tolist()))
