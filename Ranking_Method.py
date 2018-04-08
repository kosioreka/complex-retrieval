from six import iteritems
import operator
from text_preprocess import Preprocessing


class Ranking:
    PARAM_K1 = 1.2
    PARAM_B = 0.75
    EPSILON = 0.25
    DELTA = 1.0

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
            frequencies = Preprocessing.WordFrequency(document)
            self.f[idx] = frequencies

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1
        self._set_idf()

        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def _get_scores(self, query):
        scores = {}
        # for index in xrange(self.corpus_size):
        for index, doc in self.corpus.items():
            score = self._get_score(query, index)
            scores[index] = score
        return scores

    def ranked(self, queries, n):
        """Returns the `n` most relevant documents according to `query`"""
        scores = {}

        i = 1
        for query in queries:
            score = self._get_scores(query[2])
            score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
            score = score[0:n]
            scores[query[0]] = score
            progress = i / len(queries)
            print("progress:", "%.3f" % round(progress, 3))
            i += 1

        return scores
        # scores = [(index, score) for index, score in enumerate(self._get_scores(query))]
        # scores.sort(key=lambda x: x[1], reverse=True)
        # indexes, _ = self._unpack(scores)
        # return indexes[:length]

    @staticmethod
    def _unpack(tuples):
        return zip(*tuples)
