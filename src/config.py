from pathlib import Path

# hyperparameters
BATCH_SIZE = 64
SEED = 42
BLOCK_SIZE = 128
EPOCHS = 500
TRAIN_SUBSET_LENGTH = 5_000_000
TRAIN_PERC = 0.9
EVAL_PERIOD = 500
EVAL_ITERS = 200
EMBED_SIZE = 128
NUM_HEADS = 4
LEARNING_RATE = 1e-3
BLOCK_NUMBER = 4
# ----

# paths
DATA_FOLDER_PATH = Path('../data/')
DATA_FILE_PATH = DATA_FOLDER_PATH / "ubertext.wikipedia.filter_rus_gcld+short.text_only.txt"
MODELS_SAVE_PATH = Path('../weights/')
MODEL_PATH = MODELS_SAVE_PATH / 'char_bigram_b64_bs128_500epochs_ts5m_lr1e-3_4heads_emb128.pt'
TOKENIZER_PATH = Path("../data/tokenizer-ubertext-wiki.json")
TOKENIZED_DATASET_PATH = DATA_FOLDER_PATH / "ubertext.wikipedia.filter_rus_gcld+short.text_only.tokenized.bin"
# ----
