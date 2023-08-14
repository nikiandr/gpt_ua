# gpt_ua

GPT language model from scratch trained on UberText 2.0. wikipedia subcorpora.

## Train
All directions described in here were tested and written for Linux.

To train the model first you need to create virtual environment with all the necessary libs:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After creating the environment you need to download dataset. 
To do this you can run `download_data.sh` or download the **wikipedia** subcorpora split into sentences straight 
from [UberText 2.0.](https://lang.org.ua/en/ubertext/) website.

Tokenizer is included in the repository and can be used with 
HugginFace [tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) library.

To tokenize dataset and save it in binary file on disk you should run **Split and format to HF dataset** and 
**Tokenize dataset** sections in `experiments/03_tokenize_dataset.ipynb` Jupyter notebook. 

After having tokenized dataset in a file you can start training. Configs of the training process could be found in 
`src/config.py` file. To reproduce the results of this repo you should use standard config but you can change it 
to match your needs.

## Architecture 
![](./gpt_illustration.svg)

Neural network consists from 8 attention blocks.
