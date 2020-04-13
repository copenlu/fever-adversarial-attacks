import argparse
from functools import partial
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizer

NLI_DIC_LABELS = {'entailment': 2, 'neutral': 1, 'contradiction': 0}


def collate_nli_raw(instances: List[Dict],
                    tokenizer: PreTrainedTokenizer,
                    device='cuda') -> List[torch.Tensor]:
    token_ids = [tokenizer.encode(_x[0], _x[1], max_length=509) for _x in instances]
    batch_max_len = max([len(_s) for _s in token_ids])

    padded_ids_tensor = torch.tensor([_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in token_ids])

    output_tensors = [padded_ids_tensor, padded_ids_tensor > 0]

    return list(_t.to(device) for _t in output_tensors)


def collate_nli_tok_ids(instances: List[List],
                        tokenizer: PreTrainedTokenizer,
                        device='cuda') -> List[torch.Tensor]:
    batch_max_len = max([len(_s) for _s in instances])

    padded_ids_tensor = torch.tensor([_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in instances])

    output_tensors = [padded_ids_tensor, padded_ids_tensor > 0]

    return list(_t.to(device) for _t in output_tensors)


class NLIWrapper(torch.nn.Module):
    def __init__(self, model_path, device, tokenizer):
        super(NLIWrapper, self).__init__()
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.args = argparse.Namespace(**checkpoint['args'])

        transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=self.args.labels)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
        model.load_state_dict(checkpoint['model'])

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.collate_fn = partial(collate_nli_tok_ids, tokenizer=tokenizer, device=device)

    def forward(self, instances):
        self.model.eval()
        dl = DataLoader(batch_size=self.args.batch_size, dataset=instances, shuffle=False, collate_fn=self.collate_fn)
        logits_all = []
        for batch in tqdm(dl, desc="NLI Evaluation"):
            logits_val = self.model(batch[0], attention_mask=batch[1])[0]

            logits_all += logits_val.detach().cpu().numpy().tolist()

        return logits_all
