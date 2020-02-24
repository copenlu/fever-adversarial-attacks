from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import argparse
import json
import os
from tqdm import tqdm


def get_closest_sent(doc_lines, sent):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # , max_df=1.0, min_df=1
    doc_lines = [_l[0] for _l in doc_lines]
    train_X = vectorizer.fit_transform(doc_lines)
    testX = vectorizer.transform([sent]).toarray()

    cosine_similarities = linear_kernel(testX, train_X).flatten()

    related_docs_index = cosine_similarities.argsort()[:-5:-1][0]

    print(related_docs_index, cosine_similarities[related_docs_index], flush=True)

    if cosine_similarities[related_docs_index] > 0.1:
        print(doc_lines[related_docs_index], flush=True)

        return int(related_docs_index), doc_lines[related_docs_index]
    else:
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dir", help="Path to the wiki pages dump", required=True, type=str)
    parser.add_argument("--input_dataset", required=True, type=str)
    parser.add_argument("--output_dataset", required=True, type=str)
    args = parser.parse_args()

    wiki_lines = []  # wiki_id: [wiki lines]
    wiki_documents = {}

    for file in tqdm(os.scandir(args.wiki_dir), desc='Reading wiki pages...', leave=True):
        # {"id": "", "text": "", "lines": ""}
        # "lines": "0\tThe following are the football -LRB- soccer -RRB- events of the year 1928 throughout the world .\n1\t"
        with open(file) as out:
            for doc in out:
                doc = json.loads(doc)
                wiki_doc_lines = []
                for i, doc_line in enumerate(doc['lines'].split('\n')):
                    if len(doc_line.split('\t')) > 1:
                        wiki_doc_lines.append((doc_line.split('\t')[1], doc["id"], i))
                wiki_lines += wiki_doc_lines
                wiki_documents[doc['id']] = wiki_doc_lines

    title_corpus = [' '.join(line[1].split('_')) for line in wiki_lines]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # , max_df=1.0, min_df=1

    print('Building TfIdf vectorizer...', flush=True)
    train_X = vectorizer.fit_transform(title_corpus)

    with open(args.input_dataset) as out:
        for line in tqdm(out, desc='Processing dataset...', leave=True):
            line = json.loads(line)
            if line['label'] == 'NOT ENOUGH INFO':
                print(line['claim'])
                testX = vectorizer.transform([line['claim']]).toarray()

                print('Computing cosine similarities...', flush=True)
                cosine_similarities = linear_kernel(testX, train_X).flatten()
                related_docs_index = cosine_similarities.argsort()[:-5:-1]
                print(related_docs_index, cosine_similarities[related_docs_index], flush=True)

                related_docs_index = related_docs_index[0]
                if cosine_similarities[related_docs_index] > 0.5:

                    related_doc = wiki_lines[related_docs_index][1]
                    print(wiki_lines[related_docs_index], flush=True)

                    all_doc_sent = wiki_documents[related_doc]
                    sid, s = get_closest_sent(all_doc_sent, line['claim'])
                    if sid != None:
                        line['evidence'] = [['nan', 'nan', related_doc, sid, s]]

                        with open(args.output_dataset, 'a') as output:
                            output.write(json.dumps(line) + '\n')

