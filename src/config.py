from pathlib import Path

# hyperparameters
BATCH_SIZE = 32
SEED = 42
BLOCK_SIZE = 128 # 1024
EPOCHS = 7000
TRAIN_SUBSET_LENGTH = 10_000_000
TRAIN_PERC = 0.9
EVAL_PERIOD = 500
EVAL_ITERS = 100
EMBED_SIZE = 128
NUM_HEADS = 4
LEARNING_RATE = 1e-3
BLOCK_NUMBER = 4
# ----

# paths
DATA_FOLDER_PATH = Path('../data/')
DATA_FILE_PATH = DATA_FOLDER_PATH / "ubertext.wikipedia.filter_rus_gcld+short.text_only.txt"
MODELS_SAVE_PATH = Path('../weights/')
MODEL_PATH = MODELS_SAVE_PATH / (f'gpt_{BATCH_SIZE}_bs{BLOCK_SIZE}_{EPOCHS}epochs_' +
                                 f'{(TRAIN_SUBSET_LENGTH * 1e-6):.2f}mtokens_lr{LEARNING_RATE:.1e}' +
                                 f'_{NUM_HEADS}heads_emb{EMBED_SIZE}.pt')
TOKENIZER_PATH = DATA_FOLDER_PATH / 'tokenizer-ubertext-wiki.json'
TOKENIZED_DATASET_PATH = DATA_FOLDER_PATH / "ubertext.wikipedia.filter_rus_gcld+short.text_only.tokenized.npy"
HF_DATASET_PATH = DATA_FOLDER_PATH / "ubertext_wiki_sentsplit_hfdataset"
# ----

# logger
LOG_LEVEL = "DEBUG"
# ----
