# fever-adversarial-attacks
Adversarial attacks against claim detection systems at FEVER

After cloning the repo run `git submodule init` followed by `git submodule update` to pull all of the builders systems into the builders/ directory

# Dataset Preparation
`scripts/create_nli_dataset.py` can be used to create FEVER NLI dataset

# Builder Models
`builders/train_transformer.py, builders/train_rnn_cnn.py`

# Requirements:
* Glove Embeddings for the LSTM/CNN model
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
