# coding=utf-8

import math
import operator

from six import iteritems


# Implementation from https://en.wikipedia.org/wiki/Okapi_BM25
class BM25(object):
    PARAM_K1 = 1.2
    PARAM_B = 0.75
    EPSILON = 0.25

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.dl = {}
        corpus_len_sum = 0
        for k, d in corpus.items():
            self.dl[k] = float(len(d))
            corpus_len_sum += float(len(d))
        self.avgdl = corpus_len_sum / self.corpus_size
        self.corpus = corpus
        self.f = {}
        self.df = {}
        self.idf = {}
        self.average_idf = 0
        self._initialize()

    def _initialize(self):
        for idx, document in self.corpus.items():
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f[idx] = frequencies

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def _get_score(self, query, index):
        score = 0
        for word in query:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * self.average_idf
            score += (idf * self.f[index][word] * (self.PARAM_K1 + 1) / (self.f[index][word] + self.PARAM_K1 * (
                        1 - self.PARAM_B + self.PARAM_B * self.dl[index] / self.avgdl)))
        return score

    def _get_scores(self, query):
        scores = {}
        # for index in xrange(self.corpus_size):
        for index, doc in self.corpus.items():
            score = self._get_score(query, index)
            scores[index] = score
        return scores

    def ranked(self, query, length):
        """Returns the `length` most relevant documents according to `query`"""
        scores = self._get_scores(query)
        scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        return scores
        # scores = [(index, score) for index, score in enumerate(self._get_scores(query))]
        # scores.sort(key=lambda x: x[1], reverse=True)
        # indexes, _ = self._unpack(scores)
        # return indexes[:length]

    @staticmethod
    def _unpack(tuples):
        return zip(*tuples)
