import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import List
from typing import Tuple
from functools import partial
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification
from transformers import AdamW, get_constant_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data.sampler import BatchSampler
from builders.data_loader import collate_fever, FeverDataset, BucketBatchSampler, sort_key
import triggers_utils
from builders.data_loader import _LABELS as label_map
from builders.model_builder import NLILSTM
from builders.model_builder import NLICNN

from universal_triggers import attacks
import gc
from collections import defaultdict
import os
from torch.nn import DataParallel

def eval_model(model: torch.nn.Module, test_dl: BatchSampler, trigger_token_ids: List = None):
    model.eval()
    if args.labels == 2:
        loss_f = torch.nn.BCEWithLogitsLoss()
    else:
        loss_f = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        labels_all = []
        predictions_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            predictions = model(batch[0])
            loss_val = loss_f(predictions.squeeze(), batch[1])
            losses.append(loss_val.item())

            labels_all += batch[1].detach().cpu().numpy().tolist()
            predictions_all += predictions.detach().cpu().numpy().tolist()

        if args.labels == 2:
            sigmoid = torch.nn.Sigmoid()
            predictions = (sigmoid(torch.tensor(predictions_all)).squeeze() > 0.5).detach().cpu().numpy().tolist()
        else:
            predictions = np.argmax(np.array(predictions_all), axis=-1)
        # p, r, f1, _ = precision_recall_fscore_support(labels_all, predictions, average='macro')
        # print(confusion_matrix(labels_all, predictions))

        #print(confusion_matrix(labels_all, prediction))
        acc = sum(predictions == labels_all) / len(labels_all)

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--model_path", help="Path where the model will be serialized", default='ferver_bert', type=str)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)
    parser.add_argument("--target", help="The label to convert examples to", default='REFUTES', type=str)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="2 labels if NOT ENOUGH INFO excluded, 3 otherwise", type=int, default=2)
    parser.add_argument("--trigger_length", help="The total length of the trigger", type=int, default=1)
    parser.add_argument("--model", help="Model for training", type=str, default='lstm', choices=['lstm', 'cnn'])
    parser.add_argument("--embedding_dir", help="Path to directory with pretrained embeddings", default='./', type=str)
    parser.add_argument("--dropout", help="Path to directory with pretrained embeddings", default=0.1, type=float)
    parser.add_argument("--embedding_dim", help="Dimension of embeddings", choices=[50, 100, 200, 300], default=100,
                        type=int)
    # RNN ARGUMENTS
    parser.add_argument("--hidden_lstm", help="Number of units in the hidden layer", default=100, type=int)
    parser.add_argument("--num_layers", help="Number of rnn layers", default=1, type=int)
    parser.add_argument("--hidden_sizes", help="Number of units in the hidden layer", default=[100, 50], type=int,
                        nargs='+')
    # CNN ARGUMENTS
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_heights", help="filter windows", type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument("--stride", help="stride", type=int, default=1)
    parser.add_argument("--padding", help="padding", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    if not os.path.exists('./attack_results/cnn/larger_batch_size'):
        os.makedirs('./attack_results/cnn/larger_batch_size')

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=args.labels)

    print(args, flush=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    if args.model == 'lstm':
        model = NLILSTM(tokenizer, args, n_labels=args.labels).to(device)
    else:
        model = NLICNN(tokenizer, args, n_labels=args.labels).to(device)    # Adds a hook to get the embedding gradients

    triggers_utils.add_hooks_rnn_cnn(model)
    # Get the embedding weight
    embedding_weight = triggers_utils.get_embedding_weight_rnn_cnn(model)

    test = FeverDataset(args.dataset)
    # Subsample the dataset
    test.filter_dataset({args.attack_class})
    target_label = label_map[args.target]
    # print(Counter([_x['label'] for _x in test]).most_common(3))

    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                  collate_fn=collate_fn)

    checkpoint = torch.load(args.model_path)

    model.load_state_dict(checkpoint['model'])
    model = DataParallel(model).to(device)
    # Get original performance
    print("Getting original performance...")
    acc = eval_model(model, test_dl)
    print(f"Original accuracy: {acc}")

    model.train()  # rnn cannot do backwards in train mode

    # Initialize triggers
    num_trigger_tokens = args.trigger_length
    trigger_token_ids = tokenizer.convert_tokens_to_ids(["a"]) * num_trigger_tokens
    patience = 10
    count = 0
    prev_trigger = ["a"] * num_trigger_tokens
    trigger_counts = defaultdict(int)
    # sample batches, update the triggers, and repeat 3 epochs
    for _ in range(3):
        for batch in tqdm(test_dl):
            # get model accuracy with current triggers (might want to stagger this for times sake)
            # acc = eval_model(model, test_dl, trigger_token_ids)
            # print(f"Trigger: {prev_trigger}, acc: {acc}")
            model.train()  # rnn cannot do backwards in train mode

            # get grad of triggers
            averaged_grad = triggers_utils.get_average_grad_rnn_cnn(model, batch, trigger_token_ids, target_label)

            # find attack candidates using an attack method
            cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                            embedding_weight,
                                                            trigger_token_ids,
                                                            num_candidates=40)
            # cand_trigger_token_ids = attacks.random_attack(embedding_weight,
            #                                                trigger_token_ids,
            #                                                num_candidates=40)
            # cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
            #                                                        embedding_weight,
            #                                                        trigger_token_ids,
            #                                                        tree,
            #                                                        100,
            #                                                        decrease_prob=True)

            # query the model to get the best candidates
            trigger_token_ids = triggers_utils.get_best_candidates_rnn_cnn(model,
                                                          batch,
                                                          trigger_token_ids,
                                                          cand_trigger_token_ids)

            new_trigger = tokenizer.convert_ids_to_tokens(trigger_token_ids)
            trigger_counts[" ".join(new_trigger)] += 1
            #print(f"Best triggers: {new_trigger}")
            # if new_trigger == prev_trigger:
            #     count += 1
            # else:
            #     count = 0
            # if count == patience:
            #     break
            # prev_trigger = new_trigger
            gc.collect()


    # Rank all of the triggers based on the number of times they were selected
    ranked_triggers = list(sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True))
    with open(f'./attack_results/cnn/larger_batch_size/{args.attack_class}_to_{args.target}_{num_trigger_tokens}triggers.tsv', 'wt') as f:
        for trigger,count in ranked_triggers:
            trigger_token_ids = tokenizer.convert_tokens_to_ids(trigger.split(" "))
            acc = eval_model(model, test_dl, trigger_token_ids)
            f.write(f"{trigger}\t{count}\t{acc}\n")



