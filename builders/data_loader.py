import torch
from transformers import PreTrainedTokenizer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, RandomSampler, Sampler, SequentialSampler
from torch.utils.data import Dataset
import math
from typing import List, Dict
import json

_LABELS = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
_MAX_LEN_TRANSFORMER = 512 - 3  # 1 [CLS] and [2 SEP]


def identity(x):
    return x


class SortedSampler(Sampler):
    """
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/samplers/sorted_sampler.html#SortedSampler
    Samples elements sequentially, always in the same order.

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """

    def __init__(self, data, sort_key=identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/samplers/bucket_batch_sampler.py
    `BucketBatchSampler` toggles between `sampler` batches and sorted batches.
    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.
    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123)
        >>>
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size,
                 collate_fn,
                 drop_last=False,
                 shuffle=True,
                 sort_key=identity,
                 bucket_size_multiplier=100):

        self.dataset = dataset
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.collate_fn = collate_fn
        self.bucket_sampler = BatchSampler(sampler,
                                           min(batch_size * bucket_size_multiplier, len(sampler)),
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler([self.dataset[i] for i in bucket], self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield self.collate_fn([self.dataset[bucket[i]] for i in batch])

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


class FeverDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self._dataset = []
        with open(data_dir) as out:
            for line in out:
                line = json.loads(line)
                self._dataset.append(line)

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
        return self._dataset[item]


def sort_key(x):
    return len(x['claim']) + len(evidence_text(x))


def evidence_text(instance):
    return ' '.join([_s[2] for _s in instance['evidence']])


def collate_fever(instances: List[Dict], tokenizer: PreTrainedTokenizer, device='cuda') -> List[torch.Tensor]:

    token_ids = [tokenizer.encode(_x['claim'], evidence_text(_x), max_length=509) for _x in instances]
    batch_max_len = max([len(_s) for _s in token_ids])

    padded_ids_tensor = torch.tensor([_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s))
                                      for _s in token_ids]).to(device)
    labels_tensor = torch.tensor([_LABELS[_x['label']] for _x in instances], dtype=torch.long).to(device)

    return [padded_ids_tensor, labels_tensor]

