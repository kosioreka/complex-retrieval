import math
import collections
from six import iteritems
from Ranking_Method import Ranking


class TFIDFImproved(Ranking):

    def __init__(self, corpus):
        super().__init__(corpus)

    def _set_idf(self):
        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size + 1) / self.df[word]

    def _get_score(self, query, index):
        score = 0
        for word in query:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * self.average_idf
            score += idf * (1 + math.log(1 + math.log(
                self.f[index][word] / (1 - self.PARAM_B + self.PARAM_B * self.dl[index] / self.avgdl) + self.DELTA)))
        return score

    def _get_vector(self, query, index):
        score_dict = {}
        score_sum = 0
        for word in query:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * self.average_idf
            score = idf * (1 + math.log(1 + math.log(
                self.f[index][word] / (1 - self.PARAM_B + self.PARAM_B * self.dl[index] / self.avgdl) + self.DELTA)))
            score_dict[self.w_id[word]] = score
            score_sum += score
        score_dict[0] = score_sum
        score_dict = collections.OrderedDict(sorted(score_dict.items()))
        return score_dict
