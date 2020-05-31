import os
import torch
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
from argparse import Namespace

_glove_path = "glove.6B.{}d.txt".format


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def _get_glove_embeddings(embedding_dim: int, glove_dir: str):
    word_to_index = {}
    word_vectors = []

    with open(os.path.join(glove_dir, _glove_path(embedding_dim))) as fp:
        for line in tqdm(fp.readlines(), desc=f'Loading Glove embeddings {_glove_path}'):
            line = line.split(" ")

            word = line[0]
            word_to_index[word] = len(word_to_index)

            vec = np.array([float(x) for x in line[1:]])
            word_vectors.append(vec)

    return word_to_index, word_vectors


def get_embeddings(embedding_dim: int, embedding_dir: str, tokenizer: PreTrainedTokenizer):
    """
    :return: a tensor with the embedding matrix - ids of words are from vocab
    """
    word_to_index, word_vectors = _get_glove_embeddings(embedding_dim, embedding_dir)

    embedding_matrix = np.zeros((len(tokenizer), embedding_dim))

    for id in range(0, max(tokenizer.vocab.values())+1):
        word = tokenizer.ids_to_tokens[id]
        if word not in word_to_index:
            word_vector = np.random.rand(embedding_dim)
        else:
            word_vector = word_vectors[word_to_index[word]]

        embedding_matrix[id] = word_vector

    return torch.nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float), requires_grad=True)


class NLICNN(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: Namespace, n_labels: int):
        super(NLICNN, self).__init__()
        self.args = args

        self.embedding = torch.nn.Embedding(len(tokenizer), args.embedding_dim)

        self.dropout = torch.nn.Dropout(args.dropout)

        self.embedding.weight = get_embeddings(args.embedding_dim, args.embedding_dir, tokenizer)

        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(args.in_channels, args.out_channels,
                                                    (kernel_height, args.embedding_dim),
                                                    args.stride, args.padding)
                            for kernel_height in args.kernel_heights])

        output_units = n_labels if n_labels > 2 else 1
        self.final = torch.nn.Linear(len(args.kernel_heights) * args.out_channels, output_units)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input):
        input = self.embedding(input)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        input = self.dropout(input)

        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in range(len(self.conv_layers))]
        all_out = torch.cat(conv_out, 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        output = self.final(fc_in)

        return output

