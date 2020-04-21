"""
Create FEVER NLI dataset, where for each line in the original FEVER dataset with N number of evidence annotations,
we create N instances with the original fields and evidence - list of sentences # [Wikipedia URL, sentence ID, sentence]
from the wiki dump.

'NOT ENOUGH INFO' instances are discarded as they do not have any evidence annotations.
"""
import argparse
import json
import os
from tqdm import tqdm
import unicodedata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", help="Path to the wiki pages dump", required=True, type=str)
    parser.add_argument("--dataset_dirs", help="Paths to the dataset splits", required=True, type=str, nargs='+')
    parser.add_argument("--output_paths", help="Paths to serialize the datasets", required=True, type=str, nargs='+')
    args = parser.parse_args()

    wiki_docs = {}  # wiki_id: [wiki lines]

    for file in tqdm(os.scandir(args.wiki_dir), desc='Reading wiki pages...', leave=True):
        # {"id": "", "text": "", "lines": ""}
        # "lines": "0\tThe following are the football -LRB- soccer -RRB- events of the year 1928 throughout the world .\n1\t"
        with open(file) as out:
            for doc in out:
                doc = json.loads(doc)
                doc_lines = [doc_line.split('\t')[1] for doc_line in doc['lines'].split('\n')
                             if len(doc_line.split('\t')) > 1]
                wiki_docs[doc['id']] = doc_lines

    for old_dir, new_dir in zip(args.dataset_dirs, args.output_paths):
        output_writer = open(new_dir, 'w')
        with open(old_dir) as out:
            for line in tqdm(out, desc='Processing dataset...', leave=True):
                line = json.loads(line)
                if line['label'] == 'NOT ENOUGH INFO':
                    continue
                for ann in line['evidence']:
                    sentences = []
                    for ann_sent in ann:  # [Annotation ID, Evidence ID, Wikipedia URL, sentence ID]
                        wiki_url = unicodedata.normalize('NFC', ann_sent[2])
                        # [Wikipedia URL, sentence ID, sentence]
                        sentences.append([wiki_url, ann_sent[3], wiki_docs[wiki_url][ann_sent[3]]])

                    nli_line = line.copy()
                    nli_line['evidence'] = sentences
                    output_writer.write(json.dumps(nli_line)+'\n')
        output_writer.close()


# python scripts/create_nli_dataset.py --dataset_dirs data/paper_dev.jsonl data/paper_test.jsonl --wiki_dir data/wiki-pages/wiki-pages/ --output_paths data/dev_nli.jsonl data/test_nli.jsonl