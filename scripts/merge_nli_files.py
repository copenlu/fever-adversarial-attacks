"""Add 'NOT ENOUGH INFO' instances generated with the Papelo system."""
import argparse
import json
import random

if __name__ == "__main__":
    orig_files = 'data/adv_sr_nli.jsonl'
    nei_files = 'data/adv_nei_nli.jsonl'
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_file",
                        help="Path to the file with support/refute evidence-claim pairs",
                        required=True, type=str)
    parser.add_argument("--nei_file",
                        help="Path to the file with NEI evidence-claim pairs",
                        required=True, type=str)
    parser.add_argument("--output_path",
                        help="Paths to output file",
                        required=True, type=str)
    args = parser.parse_args()

    orig_lines = []
    for c in open(args.sr_file).readlines():
        json_line = json.loads(c)
        if json_line['label'].upper() in ['SUPPORTS', 'REFUTES'] and json_line['annotation'] != "N/A":
            orig_lines.append(c.strip())

    nei_lines = open(args.nei_file).readlines()

    nei_lines_cleaned = []
    for line in nei_lines:
        line_json = json.loads(line)
        if line_json['label'].upper() == 'NOT ENOUGH INFO' and line_json['annotation']!= "N/A":
            line_json['label'] = 'NOT ENOUGH INFO'
            nei_lines_cleaned.append(json.dumps(line_json))

    labels_orig = {}
    for line in orig_lines + nei_lines_cleaned:
        label = json.loads(line)['label']
        labels_orig[label] = labels_orig.get(label, 0) + 1

    print(labels_orig)
    print(len(orig_lines)+len(nei_lines_cleaned))

    final_lines = orig_lines + nei_lines_cleaned
    random.seed(42)
    random.shuffle(final_lines)

    with open(args.output_path, 'w') as out:
        out.write('\n'.join(final_lines))
