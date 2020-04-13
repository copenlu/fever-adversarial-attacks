import spacy
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd

"""
{"id": 5689, 
"verifiable": "VERIFIABLE", 
"label": "REFUTES", 
"claim": "Amyotrophic lateral sclerosis is a cure.", 
"evidence": [["Amyotrophic_lateral_sclerosis", 0, "Amyotrophic lateral sclerosis -LRB- ALS -RRB- , also known as Lou Gehrig 's disease and motor neurone disease -LRB- MND -RRB- , is a specific disease that causes the death of neurons which control voluntary muscles ."]]}

"""
nlp = spacy.load("en_core_web_sm")


def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence)
            if not (word.like_num or word.is_stop
                    or word.is_punct or word.is_space)]


def get_top_words(model, feature_names, n_top_words):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return top_words


all_tokens = []
with open('data/train_nli.jsonl') as out:
    for i, line in tqdm(enumerate(out)):
        line = json.loads(line)
        all_tokens.append(line['claim']+' '+' '.join([_l[-1] for _l in line['evidence']]))


tf_vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)
tf = tf_vectorizer.fit_transform(tqdm(all_tokens))

for comp in [10, 20, 50]:
    lda_tf = LatentDirichletAllocation(n_components=comp, random_state=0, n_jobs=-1)  # TODO: tune n_components: how many topics make sense?
    lda_tf.fit(tf)

    tfidf_feature_names = tf_vectorizer.get_feature_names()
    top_words = get_top_words(lda_tf, tfidf_feature_names, 25)

    with open(f'data/lda/top_words_{comp}', 'w') as out:
        out.write(json.dumps(top_words))

    for file in ['data/train_nli.jsonl', 'data/test_nli.jsonl', 'data/dev_nli.jsonl']:
        with open(file) as f:
            with open(f'data/lda/{file.split("/")[-1].replace(".jsonl", f"_{comp}.jsonl")}', 'w') as out:
                for i, line in tqdm(enumerate(f)):
                    line = json.loads(line)
                    text = line['claim']+' '+' '.join([_l[-1] for _l in line['evidence']])
                    doc_dist = pd.DataFrame(lda_tf.transform(tf_vectorizer.transform([text])))
                    line['topic'] = int(np.argmax(doc_dist))
                    out.write(json.dumps(line)+'\n')