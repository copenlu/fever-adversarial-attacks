import argparse
import random

import pandas as pd
from transformers import BertTokenizer
from transformers import pipeline

from attack_multiple_objectives import triggers_utils
from attack_multiple_objectives.attack_fc_nli_trans import get_fc_model, get_nli_model, get_ppl_model
from attack_multiple_objectives.nli_utils import NLI_DIC_LABELS
from builders.data_loader import BucketBatchSampler, sort_key, FeverDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--triggers_file", help="Path to the file containing adversarial triggers", type=str,
                        required=True)
    parser.add_argument("--beam_size", help="The size for beam search", type=int, default=1)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)
    parser.add_argument("--model_path", help="Path where the model will be serialized", default='ferver_bert', type=str)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--labels", help="2 labels if NOT ENOUGH INFO excluded, 3 otherwise", type=int, default=3)
    parser.add_argument("--nli_model_path", help="Path to the fine-tuned NLI model", default='snli_transformer',
                        type=str)
    args = parser.parse_args()

    device = 0 if args.gpu else -1

    # load models
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = pipeline('fill-mask', model='bert-base-uncased', device=device, topk=args.beam_size)

    nli_model, nli_model_ew, collate_nli = get_nli_model(args.nli_model_path, tokenizer, device)
    fc_model, fc_model_ew, collate_fc = get_fc_model(args.model_path, tokenizer, args.labels, device)
    ppl_model, ppl_ew = get_ppl_model(device)

    test = FeverDataset(args.dataset)
    # Subsample the dataset
    test._dataset = random.sample([i for i in test._dataset if i['label'] == args.attack_class], 200)
    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                 collate_fn=collate_fc)

    triggers = pd.read_csv(args.triggers_file, sep='\t', header=None)
    triggers.sort_values(by=2, axis=0, ascending=False)
    triggers.columns = ['trigger', 'count', 'acc', 'nli']

    print("Getting original performance...")
    orig_acc = triggers_utils.eval_model(fc_model, test_dl, labels_num=args.labels)
    print(f'Original accuracy: {orig_acc}', flush=True)

    # TODO add the GLUE functionality here

    with open(args.triggers_file + '_results', 'w') as f:
        for i, row in triggers.iterrows():
            row = row.to_dict()
            trigger = row['trigger']
            trigger_token_ids = tokenizer.convert_tokens_to_ids(trigger.split(" "))
            acc = triggers_utils.eval_model(fc_model, test_dl, trigger_token_ids, labels_num=args.labels)

            pred_dict_orig, pred_dict, prob_entail = triggers_utils.eval_nli(nli_model, test_dl, tokenizer,
                                                                             trigger_token_ids)

            ppl_loss_orig = triggers_utils.eval_ppl(ppl_model, test_dl, tokenizer)
            ppl_loss = triggers_utils.eval_ppl(ppl_model, test_dl, tokenizer, trigger_token_ids)

            delta_ppl = ppl_loss - ppl_loss_orig  # the higher, the better
            delta_acc = orig_acc - acc  # the higher, the better
            all_instances = sum([v for _, v in pred_dict_orig.items()])
            delta_ent = (pred_dict_orig[NLI_DIC_LABELS['entailment']] - pred_dict[NLI_DIC_LABELS['entailment']])
            delta_neutr = (pred_dict_orig[NLI_DIC_LABELS['neutral']] - pred_dict[NLI_DIC_LABELS['neutral']])
            delta_neg = (pred_dict_orig[NLI_DIC_LABELS['contradiction']] - pred_dict[NLI_DIC_LABELS['contradiction']])

            f.write(f"{trigger}\t{delta_acc}\t{delta_ent}\t{delta_neutr}\t{delta_neg}\t{prob_entail}\t{delta_ppl}\n")
            f.flush()
