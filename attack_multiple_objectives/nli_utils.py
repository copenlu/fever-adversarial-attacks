"""Data utils for the NLI/STS model input"""
from typing import List

import torch
from transformers import BertConfig, BertForSequenceClassification, \
    PreTrainedTokenizer

NLI_DIC_LABELS = {'entailment': 2, 'neutral': 1, 'contradiction': 0}


def collate_nli_tok_ids(instances: List[List],
                        tokenizer: PreTrainedTokenizer,
                        device='cuda') -> List[torch.Tensor]:
    batch_max_len = max([len(_s) for _s in instances])

    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in
         instances])

    output_tensors = [padded_ids_tensor, padded_ids_tensor > 0]

    return list(_t.to(device) for _t in output_tensors)
