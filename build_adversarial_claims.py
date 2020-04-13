import pandas as pd
import argparse
import numpy as np
from transformers import pipeline
from builders.data_loader import FeverDataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch
from triggers_utils import perplexity
import ipdb


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
        #for claim in tqdm(test._dataset):
        for claim in test._dataset:

            # First, find where is the best place to put the trigger
            # tloc = (-1, float('inf'))
            # tokens = tokenizer.tokenize(claim['claim'])
            # for j in range(len(tokens) - 1):
            #     if '##' not in tokens[j+1]:
            #         input_ids = [101] + tokenizer.convert_tokens_to_ids(tokens[:j] + [triggers.values[0,0]] + tokens[j:]) + [102]
            #         input_ids = torch.tensor(input_ids).unsqueeze((0)).to(device)
            #         logits = nlp.model(input_ids)[0]
            #         perp = perplexity(logits.squeeze(), input_ids.squeeze())
            #         if perp < tloc[1]:
            #             tloc = (j, perp)
            # input_ids = [101] + tokenizer.convert_tokens_to_ids(tokens + [triggers.values[0, 0]]) + [102]
            # input_ids = torch.tensor(input_ids).unsqueeze((0)).to(device)
            # logits = nlp.model(input_ids)[0]
            # perp = perplexity(logits.squeeze(), input_ids.squeeze())
            # if perp < tloc[1]:
            #     tloc = (j, perp)
            # print(tloc[1])
            # print(tokens[:tloc[0]] + [triggers.values[0, 0]] + tokens[tloc[0]:])
            # print()

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
                        # Use perplexity instead
                        pt = ' '.join(beam[0] + [token])
                        input_ids = tokenizer.encode(f"{pt} {claim['claim']}", add_special_tokens=True)
                        input_ids = torch.tensor(input_ids).unsqueeze((0)).to(device)
                        logits = nlp.model(input_ids)[0]
                        perp = perplexity(logits.squeeze(), input_ids.squeeze())
                        new_beams.append((beam[0] + [token], perp))

                        #new_beams.append((beam[0] + [token], beam[1] + np.log(probs['score'])))
                # Get new beams
                beams = sorted(new_beams, key=lambda x: x[1])[:args.beam_size]
                #beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:args.beam_size]
            prepend_text = ' '.join(beam[0])
            # Get the perplexity
            input_ids = tokenizer.encode(f"{prepend_text} {claim['claim']}", add_special_tokens=True)
            input_ids = torch.tensor(input_ids).unsqueeze((0)).to(device)
            logits = nlp.model(input_ids)[0]
            perp = perplexity(logits.squeeze(), input_ids.squeeze())
            f.write(f"{prepend_text} {claim['claim']}\t{perp}\n")
            print(f"{prepend_text} {claim['claim']}\t{perp}\n")