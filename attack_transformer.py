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
from universal_triggers import attacks
import ipdb
import gc
from collections import defaultdict
import os

def eval_model(model: torch.nn.Module, test_dl: BatchSampler, trigger_token_ids: List = None):
    model.eval()
    with torch.no_grad():
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            # Attach triggers if present
            loss, logits_val, labels = triggers_utils.evaluate_batch_bert(model, batch, trigger_token_ids)
            losses.append(loss.detach().item())

            labels_all += labels.detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.asarray(logits_all).reshape(-1, args.labels), axis=-1)
        #p, r, f1, _ = precision_recall_fscore_support(labels_all, prediction, average='binary')

        #print(confusion_matrix(labels_all, prediction))
        acc = sum(prediction == labels_all) / len(labels_all)

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

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    if not os.path.exists('./attack_results'):
        os.mkdir('./attack_results')

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=args.labels)

    print(args, flush=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
    # Adds a hook to get the embedding gradients
    triggers_utils.add_hooks_bert(model)
    # Get the embedding weight
    embedding_weight = triggers_utils.get_embedding_weight_bert(model)

    test = FeverDataset(args.dataset)
    # Subsample the dataset
    test.filter_dataset({args.attack_class})
    target_label = label_map[args.target]
    # print(Counter([_x['label'] for _x in test]).most_common(3))

    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                  collate_fn=collate_fn)

    checkpoint = torch.load(args.model_path)

    model.load_state_dict(checkpoint['model'])
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
    prev_trigger = ["a", "a"]
    trigger_counts = defaultdict(int)
    # sample batches, update the triggers, and repeat 3 epochs
    for _ in range(3):
        for batch in tqdm(test_dl):
            # get model accuracy with current triggers (might want to stagger this for times sake)
            # acc = eval_model(model, test_dl, trigger_token_ids)
            # print(f"Trigger: {prev_trigger}, acc: {acc}")
            model.train()  # rnn cannot do backwards in train mode

            # get grad of triggers
            averaged_grad = triggers_utils.get_average_grad_bert(model, batch, trigger_token_ids, target_label)

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
            trigger_token_ids = triggers_utils.get_best_candidates_bert(model,
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
    with open(f'./attack_results/{args.attack_class}_to_{args.target}_{num_trigger_tokens}triggers.tsv', 'wt') as f:
        for trigger,count in ranked_triggers[:20]:
            trigger_token_ids = tokenizer.convert_tokens_to_ids(trigger.split(" "))
            acc = eval_model(model, test_dl, trigger_token_ids)
            f.write(f"{trigger}\t{count}\t{acc}\n")



