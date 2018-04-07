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
        self.freq_dict = {}
        self.raw_data = []

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
                query_list.append((query_id, query_name, self.preprocess_text(query_name, ret="freq")))
        return query_list

    # returns each query (list entry) as a [query_id, query_name, array of words]
    def get_raw_queries(self):
        query_list = []
        for page in self.outline_pages:
            for sections in page.flat_headings_list():
                query_name = " ".join([page.page_name] + [section.heading for section in sections])
                query_id = "/".join([page.page_id] + [section.headingId for section in sections])
                query_list.append((query_id, query_name, self.preprocess_text(query_name, ret="raw")))
        return query_list

    def process_paragraphs(self):
        if not self.freq_dict:
            para_dict = {}
            raw_data = []
            with open(self.paragraph_file, 'rb') as f:
                for p in iter_paragraphs(f):
                    para_dict[p.para_id] = self.preprocess_text(p.get_text(), ret="freq")
                    raw_data.append(self.preprocess_text(p.get_text(), ret="raw"))

            self.freq_dict = para_dict
            self.raw_data = raw_data

    def get_paragraphs(self):
        self.process_paragraphs()
        return self.freq_dict

    # returns each paragraph (list entry) as an array of words
    def get_raw_paragraphs(self):
        self.process_paragraphs()
        return self.raw_data

    # preprocessing of the text
    # return [freq|raw]
    def preprocess_text(self, text: str, ret):
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
        if ret == "freq":
            freq_dict = {}
            for word in text:
                if word not in freq_dict:
                    freq_dict[word] = 0
                freq_dict[word] += 1
            return freq_dict
        elif ret == "raw":
            return text

        return None
