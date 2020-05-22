import argparse
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np

from attack_multiple_objectives import triggers_utils
from attack_multiple_objectives.attack_fc_nli_trans import get_fc_model
from builders.data_loader import BucketBatchSampler, sort_key, FeverDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action='store_true', default=False)
    parser.add_argument("--dataset", help="Path to the dataset", default='data/dev_nli.jsonl', type=str)
    parser.add_argument("--triggers_file", help="Path to the file containing adversarial triggers", type=str,
                        required=True)
    parser.add_argument("--beam_size", help="The size for beam search", type=int, default=1)
    parser.add_argument("--model_path", help="Path where the model will be serialized", default='ferver_bert', type=str)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--examples_per_trigger", help="Number of worst and best examples for a trigger", type=int, default=10)
    parser.add_argument("--labels", help="2 labels if NOT ENOUGH INFO excluded, 3 otherwise", type=int, default=3)
    parser.add_argument("--attack_class", help="The particular class to attack", default='SUPPORTS', type=str)
    parser.add_argument("--target", help="The label to convert examples to", default='REFUTES', type=str)
    parser.add_argument("--fc_model_type", help="Type of pretrained model being loaded", default='bert',
                        choices=['bert', 'roberta'])

    args = parser.parse_args()

    device = 0 if args.gpu else -1

    # load models
    if args.fc_model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    fc_model, fc_model_ew, collate_fc = get_fc_model(args.model_path, tokenizer, args.labels, device, type=args.fc_model_type)

    test = FeverDataset(args.dataset)
    test._dataset = [i for i in test._dataset if i['label'] == args.attack_class]
    test_dl = BucketBatchSampler(batch_size=args.batch_size, sort_key=sort_key, dataset=test,
                                 collate_fn=collate_fc)

    triggers = pd.read_csv(args.triggers_file, sep='\t', header=None)
    # triggers.sort_values(by=2, axis=0, ascending=False)
    triggers.columns = ['trigger', 'count']

    print("Getting original performance...")
    orig_acc = triggers_utils.eval_fc(fc_model, test_dl, labels_num=args.labels)
    print(f'Original accuracy: {orig_acc}', flush=True)

    with open(args.triggers_file + '_results', 'w') as f:
        for i, row in triggers.iterrows():
            if i > 10:
                break
            instance_losses = []
            row = row.to_dict()
            trigger = row['trigger']
            trigger_token_ids = tokenizer.convert_tokens_to_ids(trigger.split(" "))
            for batch in test_dl:
                initial_loss, _, _ = triggers_utils.evaluate_batch_bert(fc_model, batch, trigger_token_ids=None, reduction='none')
                trigger_loss, _, _ = triggers_utils.evaluate_batch_bert(fc_model, batch, trigger_token_ids=trigger_token_ids, reduction='none')
                instance_losses += (initial_loss - trigger_loss).cpu().detach().numpy().tolist()

            print('Examples...', flush=True)
            best_idx = np.argsort(instance_losses)[:args.examples_per_trigger]
            worst_idx = np.argsort(instance_losses)[-args.examples_per_trigger:]
            for idx in best_idx:
                print(f'{trigger} {test[idx]["claim"]} \t {instance_losses[idx]}', flush=True)

            small_change = []
            for l, _loss in enumerate(instance_losses):
                if int(_loss) == 0:
                    small_change.append((l, abs(_loss)))
            for idx, _loss in list(sorted(small_change, key=lambda x: x[1]))[:args.examples_per_trigger]:
                print(f'{trigger} {test[idx]["claim"]} \t {instance_losses[idx]}', flush=True)

            for idx in worst_idx:
                print(f'{trigger} {test[idx]["claim"]} \t {instance_losses[idx]}', flush=True)


"""
python attack_multiple_objectives/trigger_examples.py --examples_per_trigger 5 
--triggers_file attack_results/fc5_nli5_ppl0_SUPPORTS_to_REFUTES_1_triggers.tsv 
--gpu --model_path fever_roberta_2e5 --fc_model_type roberta --batch_size 7 --labels 3
"""