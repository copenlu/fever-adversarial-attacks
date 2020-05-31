import random
import numpy as np
import argparse
import torch
import os
import gc
import wandb

from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from datareader import GPT2FeverDataset
from datareader import collate_batch_transformer


def evaluate(model: torch.nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    with torch.no_grad():
        losses_all = []
        votes_all = []
        for batch in tqdm(dl, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            if input_ids.shape[0] != 4:
                break
            loss, logits, _ = model(input_ids, attention_mask=masks, labels=input_ids)
            losses_all.append(loss.mean().item())
    loss = np.asarray(losses_all).mean()
    print(f"Loss: {loss}")
    return loss


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--val_dataset", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--model_dir", help="Where to store the saved model", default="wandb_local", type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="l2 reg", type=float, default=0.01)
    parser.add_argument("--target_class", help="The types of claims to generate", required=True, type=str)
    parser.add_argument("--warmup_steps", help="Number of steps to warm up Adam", type=int, default=200)
    parser.add_argument("--run_name", type=str, help="A name for the run", default="pheme-baseline")
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])


    args = parser.parse_args()

    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # See if CUDA available
    device = torch.device("cpu")
    if args.n_gpu > 0 and torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    gpt2model = 'gpt2'
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    wandb.init(
        project="adversarial-fact-checking-gpt2",
        name=args.run_name,
        config={
            "epochs": n_epochs,
            "learning_rate": lr,
            "warmup": args.warmup_steps,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_split_percentage": args.train_pct,
            "seed": seed,
            "tags": ",".join(args.tags)
        }
    )

    # Create save directory for model
    if not os.path.exists(f"{args.model_dir}"):
        os.makedirs(f"{args.model_dir}")

    # Create the datareader
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2model)
    dset = GPT2FeverDataset(args.dataset_loc, tokenizer)
    # Filter to just the target class
    dset.filter_dataset({args.target_class})

    valdset = GPT2FeverDataset(args.val_dataset, tokenizer)
    # Filter to just the target class
    valdset.filter_dataset({args.target_class})

    train_dl = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_transformer
    )

    val_dl = DataLoader(
        valdset,
        batch_size=batch_size,
        collate_fn=collate_batch_transformer
    )


    # Create the model
    model = torch.nn.DataParallel(GPT2LMHeadModel.from_pretrained(gpt2model)).to(device)

    # Create the optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        n_epochs * len(train_dl)
    )

    # Train
    loss_best = evaluate(model, val_dl, device)
    for e in range(n_epochs):
        # Training loop
        for i, batch in enumerate(tqdm(train_dl)):
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            # For dataparallel issues
            if input_ids.shape[0] != batch_size:
                continue

            loss, logits, _ = model(input_ids, attention_mask=masks, labels=input_ids)

            wandb.log({"Loss": loss.mean().item()})

            loss.mean().backward()
            optimizer.step()
            scheduler.step()

        gc.collect()

        # Inline evaluation
        val_loss = evaluate(model, val_dl, device)
        wandb.log({"Val loss": val_loss})
        if val_loss < loss_best:
            best_model = model.state_dict()
            # best_loss = val_loss
            loss_best = val_loss
            torch.save(model.state_dict(), f'{args.model_dir}/model.pth')

        gc.collect()
