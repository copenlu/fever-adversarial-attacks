import argparse
import random
import numpy as np
import torch
from typing import Dict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from functools import partial
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from builders.data_loader import sort_key, BucketBatchSampler, FeverDataset, \
    collate_fever
from builders.train_utils import NLILSTM, NLICNN, EarlyStopping
from transformers import BertTokenizer


def train_model(model: torch.nn.Module,
                train_dl: BatchSampler, dev_dl: BatchSampler,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                n_epochs: int,
                early_stopping: EarlyStopping) -> (Dict, Dict):
    if args.labels == 2:
        loss_f = torch.nn.BCEWithLogitsLoss()
    else:
        loss_f = torch.nn.CrossEntropyLoss()

    best_val, best_model_weights = {'val_f1': 0}, None

    for ep in range(n_epochs):
        losses = []
        model.train()
        for i, batch in enumerate(tqdm(train_dl, desc='Training')):
            optimizer.zero_grad()
            prediction = model(batch[0])
            loss = loss_f(prediction, batch[1])
            loss.backward()

            optimizer.step()
            losses.append(loss.item())

        print('Training loss:', np.mean(losses))
        val_p, val_r, val_f1, val_loss = eval_model(model, dev_dl)
        current_val = {
            'val_p': val_p, 'val_r': val_r, 'val_f1': val_f1,
            'val_loss': val_loss, 'ep': ep
        }

        print(current_val, flush=True)

        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

        scheduler.step(val_loss)
        if early_stopping.step(val_loss):
            print('Early stopping...')
            break

    return best_model_weights, best_val


def eval_model(model: torch.nn.Module, test_dl: BucketBatchSampler):
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
            predictions = (sigmoid(torch.tensor(
                predictions_all)).squeeze() > 0.5).detach().cpu().numpy(

            ).tolist()
        else:
            predictions = np.argmax(np.array(predictions_all), axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels_all, predictions,
                                                      average='macro')
        print(confusion_matrix(labels_all, predictions))

    return p, r, f1, np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels",
                        help="2 labels if NOT ENOUGH INFO excluded, "
                             "3 otherwise",
                        type=int, default=3)

    parser.add_argument("--train_dataset", help="Path to the train datasets",
                        default='data/train_nli.jsonl', type=str)
    parser.add_argument("--dev_dataset", help="Path to the dev datasets",
                        default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--test_dataset", help="Path to the test datasets",
                        default='data/test_nli.jsonl', type=str)

    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        default='ferver_bert', type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=5e-5)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=100)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument("--patience", help="Early stopping patience", type=int,
                        default=5)

    parser.add_argument("--model", help="Model for training", type=str,
                        default='lstm', choices=['lstm', 'cnn'])

    parser.add_argument("--embedding_dir",
                        help="Path to directory with pretrained embeddings",
                        default='./', type=str)
    parser.add_argument("--dropout",
                        help="Path to directory with pretrained embeddings",
                        default=0.1, type=float)
    parser.add_argument("--embedding_dim", help="Dimension of embeddings",
                        choices=[50, 100, 200, 300], default=100, type=int)

    # RNN ARGUMENTS
    parser.add_argument("--hidden_lstm",
                        help="Number of units in the hidden layer", default=100,
                        type=int)
    parser.add_argument("--num_layers", help="Number of rnn layers", default=1,
                        type=int)
    parser.add_argument("--hidden_sizes",
                        help="Number of units in the hidden layer",
                        default=[100, 50], type=int, nargs='+')

    # CNN ARGUMENTS
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_heights", help="filter windows", type=int,
                        nargs='+', default=[2, 3, 4, 5])
    parser.add_argument("--stride", help="stride", type=int, default=1)
    parser.add_argument("--padding", help="padding", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_fever, tokenizer=tokenizer, device=device)

    if args.model == 'lstm':
        model = NLILSTM(tokenizer, args, n_labels=args.labels).to(device)
    else:
        model = NLICNN(tokenizer, args, n_labels=args.labels).to(device)

    if args.mode == 'test':
        test = FeverDataset(args.test_dataset)
        test_dl = BucketBatchSampler(batch_size=args.batch_size,
                                     sort_key=sort_key, dataset=test,
                                     collate_fn=collate_fn)

        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])

        print(eval_model(model, test_dl))
    else:
        print("Loading datasets...")
        train = FeverDataset(args.train_dataset)
        dev = FeverDataset(args.dev_dataset)

        train_dl = BucketBatchSampler(batch_size=args.batch_size,
                                      sort_key=sort_key, dataset=train,
                                      collate_fn=collate_fn)
        dev_dl = BucketBatchSampler(batch_size=args.batch_size,
                                    sort_key=sort_key, dataset=dev,
                                    collate_fn=collate_fn)

        print(model)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)
        es = EarlyStopping(patience=args.patience, percentage=False, mode='min',
                           min_delta=0.0)

        best_model_w, best_perf = train_model(model, train_dl, dev_dl,
                                              optimizer, scheduler, args.epochs,
                                              es)

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(best_perf)
        print(args)

        torch.save(checkpoint, args.model_path)
