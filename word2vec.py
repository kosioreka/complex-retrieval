import gensim
import numpy as np
from Ranking_Method import Ranking


class Word2Vec(Ranking):

    def __init__(self, training, paragraphs):
        self.paragraphs = paragraphs
        self.training = training
        self.model = self.get_model()

    def get_model(self):
        sentences = self.training
        # for document in self.corpus.items():
        #     sentences.append(document)
        model = gensim.models.Word2Vec(sentences, min_count=1)

        # idx = model.index2word
        return model

    def _get_scores(self, query):
        query_vec = sum(self.model[word] for word in query)/len(query)
        para_vec_diff = {}
        for idx, document in self.paragraphs.items():
            if len(document) == 0:
                para_vec_diff[idx] = float('inf')
                continue
            val = abs(sum(self.model[word] for word in document)/len(document) - query_vec)
            val = np.asarray(val)
            norm = np.linalg.norm(val, axis=1)
            para_vec_diff[idx] = np.sum(np.abs(val)**2, axis=-1)**(1./2)
        return para_vec_diff
