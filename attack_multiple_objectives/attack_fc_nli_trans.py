import argparse
import os
import random
from collections import defaultdict
from functools import partial

import gc
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertForMaskedLM
from transformers import BertTokenizer, BertConfig

from builders.data_loader import _LABELS as label_map
from builders.data_loader import collate_fever, FeverDataset, BucketBatchSampler, sort_key
from attack_multiple_objectives import attacks
from attack_multiple_objectives import triggers_utils
from attack_multiple_objectives.nli_utils import collate_nli_tok_ids


def get_nli_model(model_path, tokenizer, device='cpu'):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    nli_args = argparse.Namespace(**checkpoint['args'])
    transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=nli_args.labels)
    nli_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
    nli_model.load_state_dict(checkpoint['model'])
    nli_model = nli_model.to(device)
    nli_model_ew = triggers_utils.get_embedding_weight_bert(nli_model)
    collate_nli = partial(collate_nli_tok_ids, tokenizer=tokenizer, device=device)
    triggers_utils.add_hooks_bert(nli_model)

    return nli_model, nli_model_ew, collate_nli


def get_fc_model(model_path, tokenizer, labels=3, device='cpu'):
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=labels)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.train()  # rnn cannot do backwards in train mode

    triggers_utils.add_hooks_bert(model)  # Adds a hook to get the embedding gradients
    embedding_weight = triggers_utils.get_embedding_weight_bert(model)

    return model, embedding_weight, collate_fn


def get_ppl_model(device='cpu'):
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    model.train()

    triggers_utils.add_hooks_bert(model)  # Adds a hook to get the embedding gradients
    embedding_weight = triggers_utils.get_embedding_weight_bert(model)

    return model, embedding_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--model_path", help="Path where the model will be serialized", default='ferver_bert', type=str)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)
    parser.add_argument("--target", help="The label to convert examples to", default='REFUTES', type=str)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--epochs", help="Number of epochs to run trigger optimisation for.", type=int, default=3)
    parser.add_argument("--labels", help="2 labels if NOT ENOUGH INFO excluded, 3 otherwise", type=int, default=3)
    parser.add_argument("--trigger_length", help="The total length of the trigger", type=int, default=1)
    parser.add_argument("--nli_model_path", help="Path to the fine-tuned NLI model", default='snli_transformer',
                        type=str)
    parser.add_argument("--fc_w", help="The total length of the trigger", type=float, default=1.0)
    parser.add_argument("--nli_w", help="The total length of the trigger", type=float, default=0.0)
    parser.add_argument("--ppl_w", help="The total length of the trigger", type=float, default=0.0)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    print(args, flush=True)

    # load models
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    nli_model, nli_model_ew, collate_nli = get_nli_model(args.nli_model_path, tokenizer, device)
    fc_model, fc_model_ew, collate_fc = get_fc_model(args.model_path, tokenizer, args.labels, device)
    ppl_model, ppl_ew = get_ppl_model(device)

    test = FeverDataset(args.dataset)
    # Subsample the dataset
    test._dataset = random.sample([i for i in test._dataset if i['label'] == args.attack_class], 1000)
    target_label = label_map[args.target]
    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                 collate_fn=collate_fc)
    # Initialize attack_multiple_objectives
    num_trigger_tokens = args.trigger_length
    batch_triggers = []
    trigger_token_ids = tokenizer.convert_tokens_to_ids(["[MASK]"]) * num_trigger_tokens
    for e in range(args.epochs):
        trigger_counts = defaultdict(int)
        for i, batch in tqdm(enumerate(test_dl)):
            # TODO: decide if the attacks will be batch-specific
            # if e == 0:
            #     trigger_token_ids = tokenizer.convert_tokens_to_ids(["[MASK]"]) * num_trigger_tokens
            #     batch_triggers.append(trigger_token_ids)
            # else:
            #     trigger_token_ids = batch_triggers[i]

            fc_model.train()  # rnn cannot do backwards in train mode

            # get grad of attack_multiple_objectives
            # trigger_token_ids = [get_candidate_mask_tokens(batch[0], nlp, num_trigger_tokens)[0][0]]

            averaged_grad = triggers_utils.get_average_grad_bert(fc_model, batch, trigger_token_ids, target_label)
            avg_grad_nli = triggers_utils.get_average_grad_bert_nli(nli_model, batch, trigger_token_ids, tokenizer)
            avg_grad_ppl = triggers_utils.get_average_grad_bert_ppl(ppl_model, batch, trigger_token_ids, tokenizer)

            # find attack candidates using an attack method
            # cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
            #                                                 fc_model_ew,
            #                                                 trigger_token_ids,
            #                                                 num_candidates=50)
            # cand_trigger_token_ids = attacks.hotflip_attack_nli(averaged_grad, fc_model_ew,
            #                                                     avg_grad_nli, nli_model_ew,
            #                                                     num_candidates=40,
            #                                                     fc_w=0.5, nli_w=0.5)
            cand_trigger_token_ids = attacks.hotflip_attack_all(averaged_grad, fc_model_ew,
                                                                avg_grad_nli, nli_model_ew,
                                                                avg_grad_ppl, ppl_ew,
                                                                nli_w=args.nli_w, fc_w=args.fc_w, ppl_w=args.ppl_w,
                                                                num_candidates=10)

            # query the model to get the best candidates
            trigger_token_ids = triggers_utils.get_best_candidates_bert(fc_model,
                                                                        batch,
                                                                        trigger_token_ids,
                                                                        cand_trigger_token_ids)
            # batch_triggers[i] = trigger_token_ids
            new_trigger = tokenizer.convert_ids_to_tokens(trigger_token_ids)
            trigger_counts[" ".join(new_trigger)] += 1
            gc.collect()
        print(list(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))[:20], flush=True)

    # Rank all of the attack_multiple_objectives based on the number of times they were selected
    ranked_triggers = list(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))
    print(ranked_triggers, flush=True)
    os.makedirs('./attack_results', exist_ok=True)
    with open(f'./attack_results/fc{int(args.fc_w*10)}_nli{int(args.nli_w*10)}_ppl{int(args.ppl_w*10)}_{args.attack_class}_to_{args.target}_{num_trigger_tokens}_triggers.tsv',
              'wt') as f:
        for trigger, count in ranked_triggers:
            trigger_token_ids = tokenizer.convert_tokens_to_ids(trigger.split(" "))
            # acc = triggers_utils.eval_model(fc_model, test_dl, trigger_token_ids, labels_num=args.labels)
            # pred_dict_orig, pred_dict, prob_entail = triggers_utils.eval_nli(nli_model, test_dl, tokenizer, trigger_token_ids)
            f.write(f"{trigger}\t{count}\n")
