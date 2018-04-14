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
        self.corpus = []
        self.read_folds()

    def read_folds(self):
        folds_amount = range(3)#5

        for i in folds_amount:
            with open(self.qrels_file.format(nr=i), 'r') as f:
                self.true_relevance.append(load_qrels(f))
            file_reader = Preprocessing(self.outlines_file.format(nr=i), self.paragraphs_file.format(nr=i))
            self.queries.append(file_reader.get_raw_queries())
            self.paragraphs.append(file_reader.get_raw_paragraphs())
            self.corpus.extend([v for k, v in self.paragraphs[i].items()])

    # def compute(self):





def main():
    folds_train = FoldsTraining()


if __name__ == '__main__':
    main()