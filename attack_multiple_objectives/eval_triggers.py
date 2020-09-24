"""Evaluate adversarial performance of triggers."""
import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, RobertaForMaskedLM, RobertaTokenizer

from attack_multiple_objectives import triggers_utils
from attack_multiple_objectives.attack_fc_nli_trans import \
    get_checkpoint_transformer, get_fc_model
from builders.data_loader import BucketBatchSampler, FeverDataset, sort_key


def get_ppl_model(device='cpu'):
    model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
    model.train()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset",
                        default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--triggers_file",
                        help="Path to the file containing adversarial triggers",
                        type=str,
                        required=True)
    parser.add_argument("--beam_size",
                        help="The size for beam search", type=int, default=1)
    parser.add_argument("--attack_class",
                        help="The particular class to attack",
                        default='SUPPORTS', type=str)
    parser.add_argument("--model_path",
                        help="Path where the model is serialized",
                        default='ferver_roberta', type=str)
    parser.add_argument("--fc_model_type",
                        help="Type of pretrained FC model being loaded",
                        default='bert',
                        choices=['bert', 'roberta'])
    parser.add_argument("--batch_size",
                        help="Batch size", type=int, default=8)
    parser.add_argument("--labels",
                        help="2 labels if NEI excluded, 3 otherwise",
                        type=int, default=3)
    parser.add_argument("--nli_model_path",
                        help="Path to the fine-tuned NLI model",
                        default='snli_transformer',
                        type=str)

    args = parser.parse_args()

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    # load models
    if args.fc_model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    fc_model, fc_model_ew, collate_fc = get_fc_model(args.model_path, tokenizer,
                                                     args.labels, device,
                                                     type=args.fc_model_type)

    test = FeverDataset(args.dataset)
    test._dataset = [i for i in test._dataset if
                     i['label'] == args.attack_class]
    test_dl = BucketBatchSampler(batch_size=args.batch_size,
                                 sort_key=sort_key,
                                 dataset=test,
                                 collate_fn=collate_fc)

    nli_model, nli_model_ew, _, _ = \
        get_checkpoint_transformer("SparkBeyond/roberta-large-sts-b",
                                   device,
                                   hook_embeddings=True,
                                   model_type='roberta')
    nli_batch_func = partial(triggers_utils.evaluate_batch_nli,
                             tokenizer=tokenizer)

    ppl_batch_func = partial(triggers_utils.evaluate_batch_ppl,
                             tokenizer=tokenizer)
    ppl_model = get_ppl_model(device)

    triggers = pd.read_csv(args.triggers_file, sep='\t', header=None)
    triggers.columns = ['trigger','score','count']

    orig_acc = triggers_utils.eval_fc(fc_model, test_dl, labels_num=args.labels)
    print(f'Original accuracy: {orig_acc}', flush=True)
    logits_stsb = triggers_utils.eval_nli(nli_model,
                                          test_dl,
                                          tokenizer,
                                          trigger_token_ids=None)

    print(f'Original sts-b similarity class distrib: {np.mean(logits_stsb)}',
          flush=True)
    ppl_orig, ppl_std = triggers_utils.eval_ppl(ppl_model, test_dl,
                                                tokenizer,
                                                trigger_token_ids=None)
    print(f'Original ppl: {ppl_orig} {ppl_std}', flush=True)
    with open(args.triggers_file + '_results', 'w') as f:
        f.write(f'ORIGINAL\t{orig_acc:.3f}\t'
                f'{np.mean(logits_stsb)}\t'
                f'{ppl_orig:.3f} ({ppl_std:.3f})\n')
        for i, row in triggers.iterrows():
            row = row.to_dict()
            if row['count'] <= 1:
                continue
            trigger = row['trigger']
            trigger_token_ids = tokenizer.convert_tokens_to_ids(
                trigger.split(" "))
            acc = triggers_utils.eval_fc(fc_model,
                                         test_dl,
                                         trigger_token_ids,
                                         labels_num=args.labels)
            logits_stsb = triggers_utils.eval_nli(nli_model,
                                                  test_dl,
                                                  tokenizer,
                                                  trigger_token_ids)
            ppl_loss, ppl_std = triggers_utils.eval_ppl(ppl_model,
                                                        test_dl,
                                                        tokenizer,
                                                        trigger_token_ids)
            f.write(f"{trigger}\t"
                    f"{acc:.3f}\t"
                    f"{np.mean(logits_stsb)}\t"
                    f"{ppl_loss:.3f} ({ppl_std:.3f})\n")
            f.flush()
