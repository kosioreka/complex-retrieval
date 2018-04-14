from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


with open("dataset/glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

# import gensim
#
# # let X be a list of tokenized texts (i.e. list of lists of tokens)
# model = gensim.models.Word2Vec(X, size=100)
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# etree_w2v = Pipeline([
#     ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
#     ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

print("xxxx")