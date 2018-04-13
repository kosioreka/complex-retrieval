import re

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from stemming.porter2 import stem
from trec_car.read_data import iter_outlines, iter_paragraphs

GCUBE_TOKEN = "bfbfb535-3683-47c0-bd11-df06d5d96726-843339462"
DEFAULT_TAG_API = "https://tagme.d4science.org/tagme/tag"
DEFAULT_SPOT_API = "https://tagme.d4science.org/tagme/spot"
DEFAULT_REL_API = "https://tagme.d4science.org/tagme/rel"


class Preprocessing:

    def __init__(self, outline_file, paragraph_file):
        self.stop_words = stopwords.words('english')
        self.outline_file = outline_file
        self.paragraph_file = paragraph_file

        self.outline_pages = self.get_pages()
        self.freq_dict = {}
        self.raw_data = {}

        self.process_paragraphs()

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
    def get_raw_queries(self, qe_synonyms=False):
        query_list = []
        for page in self.outline_pages:
            for sections in page.flat_headings_list():
                query_name = " ".join([page.page_name] + [section.heading for section in sections])
                query_id = "/".join([page.page_id] + [section.headingId for section in sections])
                query_list.append(
                    (query_id, query_name, self.preprocess_text(query_name, ret="raw", qe_synonyms=qe_synonyms)))
        return query_list

    def process_paragraphs(self):
        if not self.freq_dict:
            para_dict = {}
            raw_data = {}
            para_text = {}
            with open(self.paragraph_file, 'rb') as f:
                for p in iter_paragraphs(f):
                    # entities = [elem.page
                    #             for elem in p.bodies
                    #             if isinstance(elem, ParaLink)]# how to retrieve entities from paragraph; p@5 get a bit higher
                    para_dict[p.para_id] = self.preprocess_text(p.get_text(), ret="freq")
                    raw_data[p.para_id] = self.preprocess_text(p.get_text(), ret="raw")
                    para_text[p.para_id] = p.get_text()

            self.freq_dict = para_dict
            self.raw_data = raw_data
            self.para_text = para_text

    def get_paragraphs(self):
        self.process_paragraphs()
        return self.freq_dict

    # returns each paragraph (list entry) as an array of words
    def get_raw_paragraphs(self):
        self.process_paragraphs()
        return self.raw_data

    # preprocessing of the text
    # return [freq|raw]
    def preprocess_text(self, text: str, ret, qe_synonyms=False):
        # mentions = tagme.mentions(text, GCUBE_TOKEN)
        # entities = " ".join([word.mention for word in mentions.get_mentions(0.01)])
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

        # adding synonyms to query
        if qe_synonyms:
            text = self.query_expansion_synonyms(text)

        # freq dictionary
        if ret == "freq":
            return Preprocessing.WordFrequency(text)
        elif ret == "raw":
            return text

        return None

    @staticmethod
    def query_expansion_synonyms(text):
        word_synonyms = set()
        for t in text:
            # print(t, "synonyms:")
            for ss in wn.synsets(t):  # Each synset represents a diff concept.
                # print(ss.definition())
                # print(ss.lemma_names())
                word_synonyms.update(ss.lemma_names())
            # print("----------------------")
        word_synonyms.update(text)
        text = word_synonyms
        text = [re.sub(r'_', ' ', t) for t in text]
        return text

    @staticmethod
    def WordFrequency(text):
        freq_dict = {}
        for word in text:
            if word not in freq_dict:
                freq_dict[word] = 0
            freq_dict[word] += 1
        return freq_dict

    def preprocess_qrels(self, qrels):
        query_id_mapping = {}
        paragraph_id_mapping = {}
        query_id_mapping_index = 0
        paragraph_id_mapping_index = 0
        query_paragraph_id_tmp = {}
        with open(qrels, 'rb') as f:
            last_query_id = None
            rank = 1
            for p in f.readlines():
                splitted = p.split()
                query_id = splitted[0].decode("utf-8")
                paragraph_id = splitted[2].decode("utf-8")
                if query_id is not None and query_id != "":
                    query_id_mapping[query_id] = query_id_mapping_index

                    if query_id != last_query_id:
                        rank = 1
                    last_query_id = query_id

                    if query_paragraph_id_tmp.get(query_id_mapping_index) is None:
                        query_paragraph_id_tmp[query_id_mapping_index] = {}

                    if paragraph_id is not None and paragraph_id != '':
                        paragraph_id_mapping[paragraph_id] = paragraph_id_mapping_index
                        query_paragraph_id_tmp[query_id_mapping_index][paragraph_id_mapping_index] = rank
                        rank += 1

                    query_id_mapping_index += 1
                    paragraph_id_mapping_index += 1


        # query_paragraph = {}
        # for query, paragraphs in query_paragraph_id_tmp.items():
        #     for paragraph_id, rank in paragraphs.items():
        #
        #         if query_paragraph.get(query) is None:
        #             query_paragraph[query] = {}
        #
        #         paragraph_array = self.raw_data[paragraph_id]
        #         paragraph_text = ' '.join(str(x) for x in paragraph_array)
        #         query_paragraph[query][paragraph_text] = rank
        print("done")
        return query_paragraph_id_tmp, query_id_mapping, paragraph_id_mapping
