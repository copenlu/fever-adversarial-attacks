import argparse
import os
import random
from collections import defaultdict
from functools import partial

import gc
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertForMaskedLM, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoModelWithLMHead
from transformers import BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer

from builders.data_loader import _LABELS as label_map
from builders.data_loader import collate_fever, FeverDataset, BucketBatchSampler, sort_key
from attack_multiple_objectives import attacks
from attack_multiple_objectives import triggers_utils
from attack_multiple_objectives.nli_utils import collate_nli_tok_ids


collate_functions = {"roberta-large-mnli": collate_nli_tok_ids,
                     "SparkBeyond/roberta-large-sts-b": collate_nli_tok_ids,
                     'roberta-large-openai-detector': collate_nli_tok_ids}

mnli_classes = {'entailment': 2, 'negation': 0, 'neutral': 1}
gpt_detector_classes = {'real': 1, 'fake': 0}


def get_checkpoint_transformer(model_name, device, hook_embeddings=False, model_type='bert'):
    """
    'roberta-large-openai-detector' output: (tensor([[-0.1055, -0.6401]], grad_fn=<AddmmBackward>),)
    Real-1, Fake-0
    Note from the authors: 'The results start to get reliable after around 50 tokens.'

    "SparkBeyond/roberta-large-sts-b" model output: (tensor([[0.6732]], grad_fn=<AddmmBackward>),)
    STS-B benchmark measure the relatedness of two sentences based on the cosine similarity of the two representations

    roberta-large-mnli output: (tensor([[-1.8364,  1.4850,  0.7020]], grad_fn=<AddmmBackward>),)
    contradiction-0, neutral-1, entailment-2
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    collate = partial(collate_functions[model_name], tokenizer=tokenizer, device=device)

    model_ew = None
    if hook_embeddings:
        model_ew = triggers_utils.get_embedding_weight_bert(model, model_type)
        triggers_utils.add_hooks_bert(model, model_type)

    return model, model_ew, collate, tokenizer


def get_fc_model(model_path, tokenizer, labels=3, device='cpu', type='bert'):
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    if type == 'bert':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=labels)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
    else:
        transformer_config = RobertaConfig.from_pretrained('roberta-base', num_labels=labels)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=transformer_config).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.train()  # rnn cannot do backwards in train mode

    triggers_utils.add_hooks_bert(model,type)  # Adds a hook to get the embedding gradients
    embedding_weight = triggers_utils.get_embedding_weight_bert(model, type)

    return model, embedding_weight, collate_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--model_path", help="Path where the model will be serialized", default='ferver_bert', type=str)
    parser.add_argument("--fc_model_type", help="Type of pretrained model being loaded", default='bert', choices=['bert', 'roberta'])
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)
    parser.add_argument("--target", help="The label to convert examples to", default='REFUTES', type=str)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--epochs", help="Number of epochs to run trigger optimisation for.", type=int, default=3)
    parser.add_argument("--labels", help="2 labels if NOT ENOUGH INFO excluded, 3 otherwise", type=int, default=3)
    parser.add_argument("--trigger_length", help="The total length of the trigger", type=int, default=1)
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
    if args.fc_model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    nli_model, nli_model_ew, nli_batch_func, gpt_model, gpt_model_ew, gpt_batch_func = None, None, None, None, None, None
    if args.nli_w > 0:
        nli_model, nli_model_ew, _, _ = \
        get_checkpoint_transformer("roberta-large-mnli", device, hook_embeddings=True, model_type='roberta')
        nli_batch_func = partial(triggers_utils.evaluate_batch_nli, tokenizer=tokenizer)

    if args.ppl_w > 0:
        gpt_model, gpt_model_ew, _, _ =\
        get_checkpoint_transformer("roberta-large-openai-detector", device, hook_embeddings=True, model_type='roberta')
        gpt_batch_func = partial(triggers_utils.evaluate_batch_gpt, tokenizer=tokenizer)

    fc_model, fc_model_ew, collate_fc = get_fc_model(args.model_path, tokenizer, args.labels, device, type=args.fc_model_type)

    test = FeverDataset(args.dataset)

    # Subsample the dataset
    test._dataset = [i for i in test._dataset if i['label'] == args.attack_class]
    target_label = label_map[args.target]
    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                 collate_fn=collate_fc)

    # Initialize attack_multiple_objectives
    num_trigger_tokens = args.trigger_length
    batch_triggers = []
    trigger_token_ids = tokenizer.convert_tokens_to_ids(["a"]) * num_trigger_tokens

    print('Dataset size', len(test._dataset), flush=True)

    previous_triggers, trigger_counts = None, None
    for e in range(args.epochs):
        trigger_counts = defaultdict(int)

        for i, batch in tqdm(enumerate(test_dl)):
            fc_model.train()  # rnn cannot do backwards in train mode

            averaged_grad = triggers_utils.get_average_grad_transformer(fc_model, batch, trigger_token_ids,
                                                                        triggers_utils.evaluate_batch_bert, target_label)
            avg_grad_nli, avg_grad_gpt = None, None
            if args.nli_w > 0:
                avg_grad_nli = triggers_utils.get_average_grad_transformer(nli_model, batch, trigger_token_ids,
                                                                           nli_batch_func, mnli_classes['entailment'])
            if args.ppl_w > 0:
                avg_grad_gpt = triggers_utils.get_average_grad_transformer(nli_model, batch, trigger_token_ids,
                                                                           gpt_batch_func, gpt_detector_classes['real'])

            cand_trigger_token_ids = attacks.hotflip_attack_all(averaged_grad, fc_model_ew,
                                                                avg_grad_nli, nli_model_ew,
                                                                avg_grad_gpt, gpt_model_ew,
                                                                nli_w=args.nli_w, fc_w=args.fc_w, ppl_w=args.ppl_w,
                                                                num_candidates=30)

            trigger_token_ids = triggers_utils.get_best_candidates_all_obj(fc_model, nli_model, gpt_model, batch,
                                                                           trigger_token_ids, cand_trigger_token_ids,
                                                                           tokenizer,
                                                                           nli_w=args.nli_w, fc_w=args.fc_w, ppl_w=args.ppl_w, beam_size=2)


            new_trigger = tokenizer.convert_ids_to_tokens(trigger_token_ids)
            trigger_counts[" ".join(new_trigger)] += 1

            gc.collect()

        best_epoch_triggers = list(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))[:20]
        if e > 3 and best_epoch_triggers[0][1] > 200:
            break

        print(best_epoch_triggers, flush=True)

    # Rank all of the attack_multiple_objectives based on the number of times they were selected
    ranked_triggers = list(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))
    print(ranked_triggers, flush=True)
    os.makedirs('./attack_results', exist_ok=True)
    with open(f'./attack_results/fc{int(args.fc_w*10)}_nli{int(args.nli_w*10)}_ppl{int(args.ppl_w*10)}'
              f'_{args.attack_class}_to_{args.target}_{num_trigger_tokens}_triggers.tsv',
              'wt') as f:
        for trigger, count in ranked_triggers:
            trigger_token_ids = tokenizer.convert_tokens_to_ids(trigger.split(" "))
            f.write(f"{trigger}\t{count}\n")
