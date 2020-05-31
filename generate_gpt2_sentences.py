import argparse
import torch
from functools import partial
#import pandas as pd
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from datareader import GPT2FeverDataset
import ipdb
import random
from itertools import zip_longest
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertForSequenceClassification
from attack_multiple_objectives.nli_utils import collate_nli_tok_ids
from transformers import BertTokenizer
from transformers import RobertaTokenizer
import numpy as np
from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification
from transformers import RobertaConfig
import attack_multiple_objectives.triggers_utils as triggers_utils
from attack_multiple_objectives.nli_utils import collate_nli_tok_ids


collate_functions = {"roberta-large-mnli": collate_nli_tok_ids,
                     "SparkBeyond/roberta-large-sts-b": collate_nli_tok_ids,
                     'roberta-large-openai-detector': collate_nli_tok_ids}


NLI_DIC_LABELS = {'SUPPORTS': 2, 'NOT ENOUGH INFO': 1, 'REFUTES': 0}

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_nli_model(model_path, tokenizer, device):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    nli_args = argparse.Namespace(**checkpoint['args'])
    transformer_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=nli_args.labels)
    nli_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=transformer_config).to(device)
    nli_model.load_state_dict(checkpoint['model'])
    nli_model = nli_model.to(device)
    collate_nli = partial(collate_nli_tok_ids, tokenizer=tokenizer, device=device)

    return nli_model, collate_nli


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
    config = RobertaConfig.from_pretrained(model_name, num_labels=1)
    model = RobertaForSequenceClassification.from_pretrained(model_name, config=config).to(device)
    collate = partial(collate_functions[model_name], tokenizer=tokenizer, device=device)

    model_ew = None
    if hook_embeddings:
        model_ew = triggers_utils.get_embedding_weight_bert(model, model_type)
        triggers_utils.add_hooks_bert(model, model_type)

    return model, model_ew, collate, tokenizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--model_loc", help="Location of the model to use", default="model.pth", type=str)
    parser.add_argument("--triggers_file", help="Location of the triggers file", default="model.pth", type=str)
    parser.add_argument("--output_file", help="Where to dave the output", type=str)
    parser.add_argument("--target_class", help="The types of claims to generate", required=True, type=str)
    parser.add_argument("--nli_model_path", help="Path to the fine-tuned NLI model", default='snli_transformer',
                        type=str)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)

    args = parser.parse_args()


    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    nli_target = NLI_DIC_LABELS[args.target_class]

    # Create the model
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(GPT2LMHeadModel.from_pretrained('gpt2')).to(device)
    model.load_state_dict(torch.load(args.model_loc))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nli_model, collate_nli = get_nli_model(args.nli_model_path, bert_tokenizer, device)

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    sts_model, sts_model_ew, _, _ = \
        get_checkpoint_transformer("SparkBeyond/roberta-large-sts-b", device, hook_embeddings=False,
                                   model_type='roberta')

    # Load the data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dset = GPT2FeverDataset(args.dataset_loc, tokenizer)
    dset.filter_dataset({args.target_class})

    # Load the triggers
    # triggers = pd.read_csv(args.triggers_file, sep='\t', header=None)
    triggers = []
    with open(args.triggers_file) as f:
        for l in f:
            triggers.append(l.strip().split('\t')[0])
            if triggers[-1][0] == 'Ġ':
                triggers[-1] = triggers[-1][1:]
    # triggers.sort_values(by=2, axis=0, ascending=False)
    # trigger_text = ','.join(triggers.values[:5,0])
    # trigger_text = "none,denied,rarely,incapable,possible"
    # trigger_text = "always,warning,proven,answer,poll"
    #trigger_text = "none"

    # print(trigger_text)
    generated_claims = []
    for s,sample in enumerate(dset._dataset):
        for trigger in grouper(triggers, 5, ''):
            trigger_text = ','.join([t.lower() for t in trigger if t != ''])
            evidence = ' '.join(r[2] for r in sample['evidence'])
            input_ids = tokenizer.encode(f"{trigger_text}||{evidence}||")
            input_ids = torch.LongTensor(input_ids).to(device)
            output = model.module.generate(input_ids.unsqueeze(0), do_sample=True, temperature=0.7, max_length=1000)
            claim_text = tokenizer.decode(output[0], skip_special_tokens=True)
            parts = claim_text.split('||')
            input_ids = bert_tokenizer.encode(parts[1], text_pair=parts[2], add_special_tokens=True, max_length=512)
            input_ids = torch.cuda.LongTensor(input_ids).unsqueeze(0)
            logits = nli_model(input_ids)[0].squeeze().cpu().detach().numpy()

            gen_claim_toks = [t.lower() for t in parts[-1].split(' ')]
            if any(t.lower() in gen_claim_toks and t != '' for t in trigger) and np.argmax(logits) == nli_target:
                print(parts[0])
                print(parts[1])
                print(parts[-1])
                # Get the STS score
                sts_input = roberta_tokenizer.encode(parts[1], text_pair=parts[2], add_special_tokens=True, max_length=512)
                sts_input = torch.cuda.LongTensor(sts_input).unsqueeze(0)
                sts_score = sts_model(sts_input)[0].squeeze().cpu().detach().item()
                print(sts_score)
                print()
                generated_claims.append(parts + [sts_score, s])
                break

    # Rank them with STS
    with open(args.output_file, 'wt') as f:
        f.write("idx\tclaim\tevidence\tsemantically coherent\tlabel\n")
        for claim in sorted(generated_claims, key=lambda x: x[-2], reverse=True):
            f.write(f"{claim[-1]}\t{claim[2]}\t{claim[1]}\t\t\n")