import argparse
import os
import random
from collections import defaultdict
from functools import partial

import gc
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    BertConfig, BertForSequenceClassification, BertTokenizer, RobertaConfig, \
    RobertaForSequenceClassification, RobertaTokenizer

from attack_multiple_objectives import attacks, triggers_utils
from attack_multiple_objectives.nli_utils import collate_nli_tok_ids
from builders.data_loader import BucketBatchSampler, FeverDataset, \
    _LABELS as label_map, collate_fever, sort_key

mnli_classes = {'entailment': 2, 'negation': 0, 'neutral': 1}
gpt_detector_classes = {'real': 1, 'fake': 0}


def get_checkpoint_transformer(model_name: str, device: str,
                               hook_embeddings: bool = False,
                               model_type: str = 'bert'):
    """
    'roberta-large-openai-detector' output: (tensor([[-0.1055, -0.6401]],
    grad_fn=<AddmmBackward>),)
    Real-1, Fake-0
    Note from the authors: 'The results start to get reliable after around 50
    tokens.'

    "SparkBeyond/roberta-large-sts-b" model output: (tensor([[0.6732]],
    grad_fn=<AddmmBackward>),)
    STS-B benchmark measure the relatedness of two sentences based on the
    cosine similarity of the two representations

    roberta-large-mnli output: (tensor([[-1.8364,  1.4850,  0.7020]],
    grad_fn=<AddmmBackward>),)
    contradiction-0, neutral-1, entailment-2
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        device)
    collate = partial(collate_nli_tok_ids, tokenizer=tokenizer, device=device)

    model_ew = None
    if hook_embeddings:
        # this allows to get the value of the gradient each time we make a
        # feed-forward pass
        model_ew = triggers_utils.get_embedding_weight_bert(model, model_type)
        triggers_utils.add_hooks_bert(model, model_type)

    return model, model_ew, collate, tokenizer


def get_fc_model(model_path, tokenizer, labels=3, device='cpu', type='bert'):
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    if type == 'bert':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=labels)
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', config=transformer_config).to(device)
    else:
        transformer_config = RobertaConfig.from_pretrained('roberta-base',
                                                           num_labels=labels)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                 config=transformer_config).to(
            device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.train()  # rnn cannot do backwards in train mode

    # Adds a hook to get the embedding gradients
    triggers_utils.add_hooks_bert(model, type)
    embedding_weight = triggers_utils.get_embedding_weight_bert(model, type)

    return model, embedding_weight, collate_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset",
                        default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--model_path",
                        help="Path where the FC model is serialized",
                        default='ferver_roberta', type=str)
    parser.add_argument("--fc_model_type",
                        help="Type of pretrained model at the model_path",
                        default='bert', choices=['bert', 'roberta'])
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--attack_class", help="The particular class to attack",
                        default='SUPPORTS', type=str)
    parser.add_argument("--target", help="The label to convert examples to",
                        default='REFUTES', type=str)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--epochs",
                        help="Number of epochs to run trigger optimisation "
                             "for.",
                        type=int, default=3)
    parser.add_argument("--labels",
                        help="2 labels if NOT ENOUGH INFO excluded, "
                             "3 otherwise",
                        type=int, default=3)
    parser.add_argument("--trigger_length",
                        help="The total length of the trigger", type=int,
                        default=1)
    parser.add_argument("--fc_w", help="The weight for the FC objective",
                        type=float, default=1.0)
    parser.add_argument("--nli_w", help="The weights for the NLI objective",
                        type=float, default=0.0)
    parser.add_argument("--ppl_w", help="The wright for the PPL objective",
                        type=float, default=0.0)
    parser.add_argument("--beam_size",
                        help="The length of the beam size for the triggers",
                        type=int, default=3)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    print(args, flush=True)

    # load models
    if args.fc_model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    nli_model, nli_model_ew, nli_batch_func, gpt_model, gpt_model_ew, \
    gpt_batch_func = None, None, None, None, None, None
    if args.nli_w > 0:
        nli_model, nli_model_ew, _, _ = \
            get_checkpoint_transformer("SparkBeyond/roberta-large-sts-b",
                                       device, hook_embeddings=True,
                                       model_type='roberta')
        nli_batch_func = partial(triggers_utils.evaluate_batch_nli,
                                 tokenizer=tokenizer)

    if args.ppl_w > 0:
        gpt_model, gpt_model_ew, _, _ = \
            get_checkpoint_transformer("roberta-large-openai-detector", device,
                                       hook_embeddings=True,
                                       model_type='roberta')
        gpt_batch_func = partial(triggers_utils.evaluate_batch_gpt,
                                 tokenizer=tokenizer)

    fc_model, fc_model_ew, collate_fc = get_fc_model(args.model_path, tokenizer,
                                                     args.labels, device,
                                                     type=args.fc_model_type)

    test = FeverDataset(args.dataset)

    # Subsample the dataset
    test._dataset = [i for i in test._dataset if
                     i['label'] == args.attack_class]
    target_label = label_map[args.target]
    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key,
                                 dataset=test,
                                 collate_fn=collate_fc)

    # Initialize attack_multiple_objectives
    num_trigger_tokens = args.trigger_length
    batch_triggers = []

    print('Dataset size', len(test._dataset), flush=True)

    batch_triggers = []
    prev_best_triggers = None
    for e in range(args.epochs):
        trigger_scores = defaultdict(int)
        trigger_counts = defaultdict(int)

        grad_accum_fc, grad_accum_nli, grad_accum_ppl = None, None, None

        for i, batch in tqdm(enumerate(test_dl)):
            if e == 0:
                batch_triggers.append(tokenizer.convert_tokens_to_ids(
                    ["[MASK]"]) * num_trigger_tokens)
            trigger_token_ids = batch_triggers[i]

            fc_model.train()  # rnn cannot do backwards in train mode

            averaged_grad = triggers_utils.get_average_grad_transformer(
                fc_model, batch, trigger_token_ids,
                triggers_utils.evaluate_batch_bert, target_label)
            if grad_accum_fc == None:
                grad_accum_fc = averaged_grad
            else:
                grad_accum_fc += averaged_grad

            avg_grad_nli, avg_grad_gpt = None, None
            if args.nli_w > 0:
                avg_grad_nli = triggers_utils.get_average_grad_transformer(
                    nli_model, batch, trigger_token_ids,
                    nli_batch_func, 5)
                if grad_accum_nli == None:
                    grad_accum_nli = avg_grad_nli
                else:
                    grad_accum_nli += avg_grad_nli

            if args.ppl_w > 0:
                avg_grad_gpt = triggers_utils.get_average_grad_transformer(
                    gpt_model, batch, trigger_token_ids,
                    gpt_batch_func, gpt_detector_classes['real'])

                if grad_accum_ppl == None:
                    grad_accum_ppl = avg_grad_gpt
                else:
                    grad_accum_ppl += avg_grad_gpt

        cand_trigger_token_ids = attacks.hotflip_attack_all(grad_accum_fc,
                                                            fc_model_ew,
                                                            grad_accum_nli,
                                                            nli_model_ew,
                                                            grad_accum_ppl,
                                                            gpt_model_ew,
                                                            nli_w=args.nli_w,
                                                            fc_w=args.fc_w,
                                                            ppl_w=args.ppl_w,
                                                            num_candidates=100)

        for i, batch in tqdm(enumerate(test_dl)):
            trigger_token_ids = batch_triggers[i]
            trigger_token_result = triggers_utils.get_best_candidates_all_obj(
                fc_model, nli_model, gpt_model, batch,
                trigger_token_ids, cand_trigger_token_ids,
                tokenizer,
                nli_w=args.nli_w, fc_w=args.fc_w,
                ppl_w=args.ppl_w, beam_size=args.beam_size)

            batch_triggers[i] = trigger_token_result[0][0]

            for _trigger in trigger_token_result:
                trigger_tokens, trigger_score = _trigger
                new_trigger = tokenizer.convert_ids_to_tokens(trigger_tokens)
                trigger_scores[" ".join(new_trigger)] += trigger_score
                trigger_counts[" ".join(new_trigger)] += 1

            gc.collect()

        best_epoch_triggers = list(
            sorted(trigger_scores.items(), key=lambda x: x[1], reverse=True))[
                              :20]

        if e == 0:
            prev_best_triggers = best_epoch_triggers[:5]
        else:
            if e > 2 and {_t[0] for _t in prev_best_triggers} == {_t[0] for _t
                                                                  in
                                                                  best_epoch_triggers[
                                                                  :5]}:
                break

            prev_best_triggers = best_epoch_triggers[:5]

        # Rank all of the attack_multiple_objectives based on the number of
        # times they were selected
        ranked_triggers = list(
            sorted(trigger_scores.items(), key=lambda x: x[1], reverse=True))
        print(ranked_triggers[:20], flush=True)
        os.makedirs('./attack_results', exist_ok=True)
        with open(f'./attack_results/fc{int(args.fc_w*10)}_'
                  f'nli{int(args.nli_w*10)}_'
                  f'ppl{int(args.ppl_w*10)}_'
                  f'{args.attack_class}_to_{args.target}_'
                  f'{num_trigger_tokens}_triggers.tsv',
                  'wt') as f:
            for trigger, score in ranked_triggers:
                trigger_token_ids = tokenizer.convert_tokens_to_ids(
                    trigger.split(" "))
                count = trigger_counts[trigger]
                f.write(f"{trigger}\t{score}\t{count}\n")
