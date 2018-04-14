import operator

from TF_IDF_Improved import TFIDFImproved
from eval_framework import load_qrels
from text_preprocess import Preprocessing


class FoldsTraining(object):

    def __init__(self, file_name='', directory_name=''):
        if directory_name is not '':
            self.directory_name = directory_name + '/'
        else:
            self.directory_name = 'benchmarkY1-train/'
        if file_name is not '':
            self.file_name = file_name
        else:
            self.file_name = 'train.pages.cbor'
        self.qrels_file = self.directory_name + 'fold-{nr}-' + self.file_name + '-hierarchical.qrels'
        self.outlines_file = self.directory_name + 'fold-{nr}-' + self.file_name + '-outlines.cbor'
        self.paragraphs_file = self.directory_name + 'fold-{nr}-' + self.file_name + '-paragraphs.cbor'
        self.true_relevance = []
        self.queries = []
        self.paragraphs = []
        self.corpus = {}
        self.read_folds()
        self.write_file('train', 0)
        self.write_file('test', 1)

    def read_folds(self):
        folds_amount = range(2)  # 5

        for i in folds_amount:
            with open(self.qrels_file.format(nr=i), 'r') as f:
                self.true_relevance.append(load_qrels(f))
            file_reader = Preprocessing(self.outlines_file.format(nr=i), self.paragraphs_file.format(nr=i))
            self.queries.append(file_reader.get_raw_queries())
            self.paragraphs.append(file_reader.get_raw_paragraphs())
            self.corpus.update(self.paragraphs[i])
        self.tfidf = TFIDFImproved(self.corpus)


    def run_tfidf(self, queries_dict):
        scores = self.tfidf.ranked_vector(queries_dict, 10)
        return scores

    def write_file(self, file_name, fold_nr):
        scores = self.run_tfidf(self.queries[fold_nr])
        bad_para = 0
        with open(file_name + ".txt", "w") as text_file:
            for q_id, doc_dict in scores.items():
                if q_id not in self.true_relevance[fold_nr]:
                    bad_para += 1
                    continue
                doc_dict_sort = sorted(doc_dict.items(), key=lambda x: x[1][0], reverse=True)
                counter = 0
                add = True
                for elem in doc_dict_sort:
                    d_id = elem[0]
                    word_dict = elem[1]
                    if d_id in self.true_relevance[fold_nr][q_id]:
                        rel = 1
                        add = True
                        counter += 1
                    else:
                        rel = 0
                        if counter > 10:
                            add = False
                        counter += 1
                    if add:
                        line = "{r} qid:{q}".format(r=rel, q=q_id)
                        for w_id, val in word_dict.items():
                            if w_id is not 0:
                                line += " {w}:{v}".format(w=w_id, v=val)
                        line += " #docid = {d} score = {s}".format(d=d_id, s=word_dict[0])
                        print(line, file=text_file)
                        # doc_dict_sort.remove(rd_id)
        print("Bad para: ", bad_para)



def main():
    folds_train = FoldsTraining()


if __name__ == '__main__':
    main()
