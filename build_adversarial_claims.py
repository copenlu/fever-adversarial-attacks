import pandas as pd
import argparse
import numpy as np
from transformers import pipeline
from builders.data_loader import FeverDataset
from transformers import BertTokenizer
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--triggers_file", help="Path to the file containing adversarial triggers", type=str, required=True)
    parser.add_argument("--beam_size", help="The size for beam search", type=int, default=1)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)

    args = parser.parse_args()

    device = 0 if args.gpu else -1

    nlp = pipeline('fill-mask', model='bert-base-uncased', device=device, topk=args.beam_size)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the triggers
    triggers = pd.read_csv(args.triggers_file, sep='\t', header=None)
    triggers.sort_values(by=2, axis=0, ascending=False)
    # Load the dataset
    test = FeverDataset(args.dataset)
    # Subsample the dataset
    test.filter_dataset({args.attack_class})

    with open(f'claims_{args.attack_class}.txt', 'wt') as f:
        for claim in tqdm(test._dataset):
            # TODO: come up with a better stopping criterion
            beams = [([triggers.values[i,0]], 0) for i in range(args.beam_size)]
            while len(beams[0][0]) < 5:
                new_beams = []
                for beam in beams:
                    prepend_text = ' '.join(beam[0])
                    most_probable = nlp(f"{prepend_text} [MASK] {claim['claim']}")
                    # Get the log probability of each
                    for probs in most_probable:
                        token = tokenizer.convert_ids_to_tokens(probs['token'])
                        new_beams.append((beam[0] + [token], beam[1] + np.log(probs['score'])))
                # Get new beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:args.beam_size]
            prepend_text = ' '.join(beam[0])
            f.write(f"{prepend_text} {claim['claim']}\n")