from xml.dom import minidom
from typing import AnyStr
from typing import List
from typing import Tuple
from typing import Set
import unicodedata
import json
import random
import string

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


stopwords_en = set(stopwords.words('english'))
punc = set(string.punctuation)
with open('./data/indomain_words.txt') as f:
    eng_words = set([l.strip().lower() for l in f])


def text_to_batch_transformer(claims: List, tokenizer: PreTrainedTokenizer, evidence: List) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :param: text_pair: An optional second string (for multiple sentence sequences)
    :return: A list of IDs and a mask
    """
    # Create the input string; first get a target word
    cands = [[w for w in word_tokenize(c) if w not in stopwords_en and w not in punc] for c in claims]
    #targets = [','.join([w.lower() for w in set(random.sample(cand, min(1,len(cand))) + random.sample(eng_words, 4))]) for cand in cands]
    # Using only candidates
    targets = [','.join([w.lower() for w in set(random.sample(cand, min(5,len(cand))))])for cand in cands]
    # # First get 5 possibel real candidates and 25 noise candidates
    # potential_words = [[w.lower() for w in set(random.sample(cand, min(5,len(cand))) + random.sample(eng_words, 25))] for cand in cands]
    # # Now randomly select 5 words from this list; we add more possible noise to give the model a better chance at generating good claims
    # # we want the model to just add words when they make sense, not force it to always pick 1-2 words
    # targets = [','.join(random.sample(pw, 5)) for pw in potential_words]
    texts = [f"{target}||{evid}||{claim}" for target,evid,claim in zip(targets, evidence, claims)]
    input_ids = [tokenizer.encode(t, max_length=tokenizer.max_len-1) + [tokenizer.eos_token_id] for t in texts]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks)


def collate_batch_transformer_with_index(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    return collate_batch_transformer(input_data) + ([i[-1] for i in input_data],)


class GPT2FeverDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self._dataset = []
        self.tokenizer = tokenizer
        with open(data_dir) as out:
            for line in out:
                line = json.loads(line)
                self._dataset.append(line)

    def filter_dataset(self, labels: Set[AnyStr]):
        """
        Filters the dataset to only samples with the provided labels
        :param labels: A list of valid labels
        """
        self._dataset = [ex for ex in self._dataset if ex['label'] in labels]


    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        """
        :return:
        {'id': 163803,
        'verifiable': 'VERIFIABLE',
        'label': 'SUPPORTS',
        'claim': 'Ukrainian Soviet Socialist Republic was a founding participant of the UN.',
        'evidence': [['Ukrainian_Soviet_Socialist_Republic', 7,
            'The Ukrainian SSR was a founding member of the United Nations , although it was legally represented by the
            All-Union state in its affairs with countries outside of the Soviet Union .'
        ]]}
        """
        row = self._dataset[item]
        claim = row['claim']
        evidence = ' '.join(r[2] for r in row['evidence'])
        input_ids, masks = text_to_batch_transformer([claim], self.tokenizer, [evidence])
        return input_ids, masks, item