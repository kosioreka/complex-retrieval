from eval_framework import load_qrels
from text_preprocess import Preprocessing


class FoldsTraining(object):

    def __init__(self, file_name='', directory_name='', nb_folds=2):
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
        self.true_relevance_stemmed = []
        self.queries = []
        self.queries_all = []
        self.paragraphs = []
        self.paragraphs_dict = {}
        self.corpus = []
        self.read_folds(nb_folds)

    def read_folds(self, nb_folds=2):
        folds_amount = range(nb_folds)  # 5

        for i in folds_amount:
            file_reader = Preprocessing(self.outlines_file.format(nr=i), self.paragraphs_file.format(nr=i))
            with open(self.qrels_file.format(nr=i), 'r') as f:
                qrels = load_qrels(f)
                self.true_relevance.append(qrels)
                self.true_relevance_stemmed.append(file_reader.preprocess_stem_qrels(qrels))
            self.queries.append(file_reader.get_raw_queries(qe_synonyms=False))
            self.queries_all.extend(file_reader.get_raw_queries())
            self.paragraphs.append(file_reader.get_raw_paragraphs())
            self.paragraphs_dict.update(file_reader.get_raw_paragraphs())
            self.corpus.extend([v for k, v in self.paragraphs[i].items()])

    # def compute(self):


def main():
    folds_train = FoldsTraining()
    tr = folds_train.true_relevance_stemmed
    print("X")


if __name__ == '__main__':
    main()
