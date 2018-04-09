import argparse

from BM25 import BM25
from TF_IDF_Improved import TFIDFImproved
from text_preprocess import Preprocessing
from trec_car.format_runs import *


def run_bm25(queries_dict, paragraphs_dict):
    bm25 = BM25(paragraphs_dict)
    scores = bm25.ranked(queries_dict, 10)
    return output_results(scores)


def run_tfidf(queries_dict, paragraphs_dict):
    tfidf = TFIDFImproved(paragraphs_dict)
    scores = tfidf.ranked(queries_dict, 10)
    return output_results(scores)


def output_results(scores):
    output_entries = []
    for query_id, score in scores.items():
        rank = 1
        for paragraph_score in score:
            entry = RankingEntry(query_id, paragraph_score[0], rank, paragraph_score[1])
            output_entries.append(entry)
            rank += 1

    return output_entries


def main():
    args = parse_arguments()
    preprocessing = Preprocessing(args.outline_file, args.paragraph_file)
    queries_dict = preprocessing.get_raw_queries(qe_synonyms=True)
    paragraphs_dict = preprocessing.get_raw_paragraphs()

    # output_entries = run_bm25(queries_dict, paragraphs_dict)
    # save_scores_to_file(output_entries, "bm25_synonyms.out")

    output_entries = run_tfidf(queries_dict, paragraphs_dict)
    save_scores_to_file(output_entries, "tfidf_synonyms.out")

    #testing
    paragraph = preprocessing.para_text[output_entries[0].paragraph_id]
    print("Query:", output_entries[0].query_id, "paragraph:\n", paragraph)


    print('end')


def save_scores_to_file(output_entries, filename="test.out"):
    with open(filename, mode='w', encoding='UTF-8') as f:
        writer = f
        temp_list = []
        for entry in output_entries:
            temp_list.append(entry)
        format_run(writer, temp_list, exp_name='test')
        f.close()


def parse_arguments():
    parser = argparse.ArgumentParser(prog='complex-retrieval', description='Complex Answer Retrieval')
    parser.add_argument("outline_file", type=str, help="Location of the outline file", action='store')
    parser.add_argument("paragraph_file", type=str, help="Location of the paragraph file", action='store')

    args = parser.parse_args()
    return args


main()
