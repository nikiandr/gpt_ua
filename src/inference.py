import argparse
from pathlib import Path
from tokenizers import Tokenizer
import torch

parser = argparse.ArgumentParser(description='Inference of GPT network.')

parser.add_argument('weights', metavar='<weights>', type=str,
                    help='path to model weights')
parser.add_argument('tokenizer_path', metavar='<tokenizer_path>', type=str,
                    help='tokenizer file path')
parser.add_argument('start', metavar='<start>', type=str,
                    help='start of string for generating')
parser.add_argument('--num_tokens', type=int, default=100,
                    help='number of tokens to generate (default: 100)')
parser.add_argument('--device', type=str, choices=["cuda", "cpu"],
                    help='device on which to run inference (default: cuda if available, else cpu)')

args = parser.parse_args()

match args.device:
    case "cuda":
        DEVICE = "cuda" if torch.cuda.is_available() else None
        if DEVICE is None:
            raise ValueError("Device cuda is not available.")
    case "cpu":
        DEVICE = "cpu"
    case _:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer_path = Path(args.tokenizer_path)
if not tokenizer_path.is_file() or tokenizer_path.suffix != ".json":
    raise ValueError(f"Tokenizer in wrong format or file doesn't exist: [{tokenizer_path}]")
tokenizer = Tokenizer.from_file(str(tokenizer_path))

weights_path = Path(args.weights)
if not weights_path.is_file() or weights_path.suffix != ".pt":
    raise ValueError(f"Model weights in wrong format or file doesn't exist: [{weights_path}]")

model = torch.load(weights_path).to(DEVICE)
model.eval()

starting_string = args.start

start_encoded = tokenizer.encode(starting_string).ids

idx = torch.tensor(start_encoded, dtype=torch.long).reshape(1, -1).to(DEVICE)
idx_gen = model.generate(idx, max_token_gen=args.num_tokens)
print(f"Generating example: {starting_string}" + tokenizer.decode(idx_gen[0].tolist()))
