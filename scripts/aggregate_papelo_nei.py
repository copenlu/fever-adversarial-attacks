"""
Add 'NOT ENOUGH INFO' instances generated with the Papelo system.
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

    for papelo_path, new_path in zip(args.dataset_dirs, args.output_paths):
        """{"id": 56204, "verifiable": "NOT VERIFIABLE", "label": "NOT ENOUGH INFO", 
        "claim": "Keith Urban is a person who sings.", 
        "evidence": [[[176603, null, null, null], [178904, null, null, null], [313935, null, null, null]]], 
        "predicted_pages": ["Keith_Urban_-LRB-1999_album-RRB-", "Days_Go_By"], 
        "predicted_sentences": [["Keith_Urban_-LRB-1999_album-RRB-", 0], ["Days_Go_By", 0]]}
        """
        output_writer = open(new_path, 'w')
        with open(papelo_path) as out:
            for line in tqdm(out, desc='Processing dataset...', leave=True):
                line = json.loads(line)
                for predicted_sent in line['predicted_sentences'][:1]:
                    wiki_url = unicodedata.normalize('NFC', predicted_sent[0])
                    sent_id = predicted_sent[1]
                    if len(wiki_docs[wiki_url]) > sent_id:
                        sentence = wiki_docs[wiki_url][sent_id]
                        sentences = [[wiki_url, sent_id, sentence]]
                        nli_line = line.copy()
                        nli_line['evidence'] = sentences
                        del nli_line['predicted_pages']
                        del nli_line['predicted_sentences']
                        output_writer.write(json.dumps(nli_line)+'\n')
        output_writer.close()

#python scripts/aggregate_papelo_nei.py --dataset_dirs data/fever/train_nei.sentences.p5.s5.jsonl --wiki_dir data/wiki-pages/wiki-pages/ --output_paths data/adv_nei_nli.jsonl
