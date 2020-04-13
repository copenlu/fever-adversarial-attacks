import argparse
import pandas as pd
from builders.data_loader import FeverDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)
    parser.add_argument("--target", help="The label to convert examples to", default='REFUTES', type=str)
    parser.add_argument("--trigger_length", help="The total length of the trigger", type=int, default=1)
    parser.add_argument("--trigger_loc", help="Directory where the triggers files are located", type=str, required=True)
    parser.add_argument("--n_triggers", help="Number of different triggers to print", type=int, default=1)
    args = parser.parse_args()

    print(args, flush=True)

    test = FeverDataset(args.dataset)
    # Subsample the dataset
    test.filter_dataset({args.attack_class})
    unique_claims = set([s['claim'] for s in test._dataset])

    triggers = pd.read_csv(
        f"{args.trigger_loc}/{args.attack_class}_to_{args.target}_{args.trigger_length}triggers.tsv",
        sep='\t',
        header=None
    )
    # Sort by accuracy
    triggers.sort_values(by=2, inplace=True)
    for j,row in enumerate(triggers.values[:args.n_triggers]):
        with open(f"{args.trigger_loc}/{args.attack_class}_to_{args.target}_{args.trigger_length}triggers_statements_{j}.txt", 'wt') as f:
            for sample in unique_claims:
                f.write(' '.join([row[0], sample]) + '\n')


