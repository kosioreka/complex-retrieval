from gensim.models import Word2Vec
from vec4ir.word2vec import WordCentroidDistance, WordMoversDistance
from vec4ir.core import Retrieval, all_but_the_top
from vec4ir.base import Tfidf, Matching
from text_preprocess import Preprocessing
import numpy as np

preprocessing = Preprocessing("test200.v2.0\\all.test200.cbor.outlines", "test200.v2.0\\all.test200.cbor.paragraphs")
# preprocessing = Preprocessing(args.outline_file, args.paragraph_file)
queries_dict = preprocessing.get_raw_queries(qe_synonyms=False)
paragraphs_dict = preprocessing.get_raw_paragraphs()

documents = []
for k, v in paragraphs_dict.items():
    for word in v:
        documents.append(word)

match_op = Matching()
model = Word2Vec(documents, min_count=1)
wcd = WordCentroidDistance(model)
RM = Retrieval(wcd, matching=match_op)
RM.fit(documents)

X = []
Y, query_id_mapping, paragraph_id_mapping_index = preprocessing.preprocess_qrels("test200.v2.0\\all.test200.cbor.hierarchical.qrels")
y_queries = [y_queries for y_queries, v in Y.items()]

for query in queries_dict:
    if query_id_mapping.get(query[0]) is not None:
        X.append([query_id_mapping[query[0]], query[1]])

scores = RM.evaluate(X, Y)
mean_scores = {k : np.mean(v) for k,v in scores.items()}
print(mean_scores)