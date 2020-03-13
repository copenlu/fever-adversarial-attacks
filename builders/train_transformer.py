import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict
from functools import partial
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification
from transformers import AdamW, get_constant_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data.sampler import BatchSampler
from builders.data_loader import collate_fever, FeverDataset, BucketBatchSampler, sort_key


def train_model(model: torch.nn.Module,
                train_dl: BatchSampler, dev_dl: BatchSampler,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                n_epochs: int) -> (Dict, Dict):

    best_val, best_model_weights = {'val_f1': 0}, None

    for ep in range(n_epochs):
        for i, batch in enumerate(tqdm(train_dl, desc='Training')):
            model.train()
            optimizer.zero_grad()
            loss, logits = model(batch[0], attention_mask=batch[0]>1, labels=batch[1])

            loss.backward()

            optimizer.step()
            scheduler.step()

        val_p, val_r, val_f1, val_loss = eval_model(model, dev_dl)
        current_val = {'val_f1': val_f1, 'val_p': val_p, 'val_r': val_p, 'val_loss': val_loss, 'ep': ep}
        print(current_val, flush=True)

        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

    return best_model_weights, best_val


def eval_model(model: torch.nn.Module, test_dl: BatchSampler):
    model.eval()
    loss_f = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            loss, logits_val = model(batch[0], attention_mask=batch[0]>1, labels=batch[1])
            loss = loss_f(logits_val, batch[1].long())
            losses.append(loss.item())

            labels_all += batch[1].detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.asarray(logits_all).reshape(-1, args.labels), axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels_all, prediction, average='macro')
        print(confusion_matrix(labels_all, prediction))

    return p, r, f1, np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="2 labels if NOT ENOUGH INFO excluded, 3 otherwise", type=int, default=3)

    parser.add_argument("--train_dataset", help="Path to the train datasets", default='data/train_nli.jsonl', type=str)
    parser.add_argument("--dev_dataset", help="Path to the dev datasets", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--test_dataset", help="Path to the test datasets", default='data/test_nli.jsonl', type=str)

    parser.add_argument("--model_path", help="Path where the model will be serialized", default='ferver_bert', type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=5e-5)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str, default='train', choices=['train', 'test'])

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=args.labels)

    print(args, flush=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    if args.mode == 'test':
        test = FeverDataset(args.test_dataset)
        # print(Counter([_x['label'] for _x in test]).most_common(3))

        test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                      collate_fn=collate_fn)

        checkpoint = torch.load(args.model_path)

        model.load_state_dict(checkpoint['model'])
        print(eval_model(model, test_dl))

    else:
        print("Loading datasets...")
        train = FeverDataset(args.train_dataset)
        dev = FeverDataset(args.dev_dataset)

        # print(Counter([_x['label'] for _x in train]).most_common(3))
        # print(Counter([_x['label'] for _x in dev]).most_common(3))

        train_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=train,
                                      collate_fn=collate_fn)
        dev_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=dev, collate_fn=collate_fn)

        num_train_optimization_steps = int(args.epochs * len(train) / args.batch_size)

        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0.05)

        best_model_w, best_perf = train_model(model, train_dl, dev_dl, optimizer, scheduler, args.epochs)

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(best_perf)
        print(args)

        torch.save(checkpoint, args.model_path)
