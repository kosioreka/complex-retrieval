import re
from nltk.corpus import stopwords
from trec_car.read_data import iter_outlines, iter_paragraphs
from stemming.porter2 import stem


class Preprocessing:

    def __init__(self, outline_file, paragraph_file):
        self.stop_words = stopwords.words('english')
        self.outline_file = outline_file
        self.paragraph_file = paragraph_file

        self.outline_pages = self.get_pages()

    def get_pages(self):
        with open(self.outline_file, 'rb') as f:
            pages = [p for p in iter_outlines(f)]
        return pages

    def get_queries(self):
        query_list = []
        for page in self.outline_pages:
            for sections in page.flat_headings_list():
                query_name = " ".join([page.page_name] + [section.heading for section in sections])
                query_id = "/".join([page.page_id] + [section.headingId for section in sections])
                query_list.append((query_id, query_name, self.preprocess_text(query_name)))
        return query_list

    def get_paragraphs(self):
        para_dict = {}
        with open(self.paragraph_file, 'rb') as f:
            for p in iter_paragraphs(f):
                para_dict[p.para_id] = self.preprocess_text(p.get_text())
        return para_dict

    # preprocessing of the text
    def preprocess_text(self, text: str):
        # TODO: annotations?
        # lower case
        text = text.lower()
        # special characters removal
        text = re.sub('[^A-Za-z0-9 \n]+', '', text)
        # stop words removal
        text = [word for word in text.split() if word not in self.stop_words]
        # stemming
        text = [stem(word) for word in text]
        # TODO: Query expansion for limitations like: irregular verbs, synonyms...?
        # freq dictionary
        freq_dict = {}
        for word in text:
            if word not in freq_dict:
                freq_dict[word] = 0
            freq_dict[word] += 1
        return freq_dict
