from gensim.models import Word2Vec
from vec4ir.word2vec import WordCentroidDistance, WordMoversDistance
from vec4ir.core import Retrieval, all_but_the_top
from vec4ir.base import Tfidf, Matching
from text_preprocess import Preprocessing


preprocessing = Preprocessing("test200.v2.0\\all.test200.cbor.outlines", "test200.v2.0\\all.test200.cbor.paragraphs")
# preprocessing = Preprocessing(args.outline_file, args.paragraph_file)
queries_dict = preprocessing.get_raw_queries(qe_synonyms=False)
paragraphs_dict = preprocessing.get_raw_paragraphs()

documents = []
for k, v in paragraphs_dict.items():
    documents.append(v)


match_op = Matching()
model = Word2Vec(documents, min_count=1)
wcd = WordCentroidDistance(model)
RM = Retrieval(wcd, matching=match_op).fit(documents)
