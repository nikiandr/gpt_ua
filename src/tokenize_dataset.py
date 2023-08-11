from tokenizers import Tokenizer
import numpy as np
from config import TOKENIZER_PATH, DATA_FILE_PATH, TOKENIZED_DATASET_PATH

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
with open(DATA_FILE_PATH) as f:
    train_subset = f.read()
train_encoded = np.array(tokenizer.encode(train_subset).ids, dtype=np.long)
np.save(TOKENIZED_DATASET_PATH, train_encoded)
