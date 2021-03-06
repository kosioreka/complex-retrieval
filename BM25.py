# coding=utf-8

import math
from six import iteritems

# Implementation from https://en.wikipedia.org/wiki/Okapi_BM25
from Ranking_Method import Ranking


class BM25(Ranking):

    def __init__(self, corpus):
        super().__init__(corpus)

    def _set_idf(self):
        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def _get_score(self, query, index):
        score = 0
        for word in query:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * self.average_idf
            score += (idf * self.f[index][word] * (self.PARAM_K1 + 1) / (self.f[index][word] + self.PARAM_K1 * (
                    1 - self.PARAM_B + self.PARAM_B * self.dl[index] / self.avgdl)))
        return score
