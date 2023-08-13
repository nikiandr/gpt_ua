import time
from tqdm.auto import tqdm
from utils import TqdmLoggingHandler
import os
import logging

import torch
import model as bg
import numpy as np
from tokenizers import Tokenizer
from torchinfo import summary

from config import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
logger.addHandler(TqdmLoggingHandler())

logger.debug(f"Current folder: {os.getcwd()}")
logger.debug(f"Current device: {DEVICE}")

torch.manual_seed(SEED)

logger.debug(f"Memory allocated before start: {torch.cuda.memory_allocated() * 1e-9} gb")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
train_subset = np.load(TOKENIZED_DATASET_PATH, mmap_mode='r')[:TRAIN_SUBSET_LENGTH]

train_num = int(TRAIN_PERC * len(train_subset))
train, val = train_subset[:train_num], train_subset[train_num:]

train_tensor = torch.tensor(train, dtype=torch.long)
val_tensor = torch.tensor(val, dtype=torch.long)

logger.debug(f"Train tensor shape: {train_tensor.size()}")
logger.debug(f"Train tensor shape: {val_tensor.size()}")

# print(f"Memory allocated after dataset and tokenizer loading: {torch.cuda.memory_allocated() * 1e-9} gb")

# ex-model place

model = bg.GPT(dict_size=tokenizer.get_vocab_size(),
               embedding_size=EMBED_SIZE,
               block_size=BLOCK_SIZE,
               num_heads=NUM_HEADS,
               block_number=BLOCK_NUMBER).to(DEVICE)


if logger.level == logging.DEBUG:
    xb, _ = bg.random_batch(train_tensor, batch_s=BATCH_SIZE, bls=BLOCK_SIZE)
    xb = xb.to(DEVICE)
    logger.debug(f"{summary(model, input_data=xb)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

time.sleep(1)
# print(f"Memory allocated after model creating: {torch.cuda.memory_allocated() * 1e-9} gb")

for epoch in tqdm(range(EPOCHS)):
    if epoch % EVAL_PERIOD == 0:
        losses = model.estimate_loss(train_set=train_tensor,
                                     val_set=val_tensor,
                                     batch_size=BATCH_SIZE,
                                     eval_iters=EVAL_ITERS)
        logger.info(f"Iteration: {epoch}, validation loss: {losses['val']}, train loss: {losses['train']}")
    logger.debug(f"\nMemory allocated after batching on epoch {epoch}: {torch.cuda.memory_allocated() * 1e-9} gb\n")
    xb, yb = bg.random_batch(train_tensor, batch_s=BATCH_SIZE, bls=BLOCK_SIZE)
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    logger.debug(f"Memory allocated after batching on epoch {epoch}: {torch.cuda.memory_allocated() * 1e-9} gb")
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
logger.info(f"Final loss: {loss.item()}")
torch.save(model, MODEL_PATH)


idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
idx_gen = model.generate(idx, max_token_gen=100)
logger.info("Generating example: Українська мова це " + tokenizer.decode(idx_gen[0].tolist()))
