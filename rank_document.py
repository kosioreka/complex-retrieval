import argparse
import operator
from trec_car.format_runs import *

from BM25_2 import BM25
from text_preprocess import Preprocessing


def main():
    args = parse_arguments()
    preprocessing = Preprocessing(args.outline_file, args.paragraph_file)
    queries_dict = preprocessing.get_raw_queries()
    paragraphs_dict = preprocessing.get_raw_paragraphs()

    queries_list = queries_dict[0: 1]
    bm25 = BM25(paragraphs_dict)
    query = queries_list[0][2]
    scores = bm25.ranked(query, 10)

    rank = 1
    for score in scores:
        entry = RankingEntry(queries_list[0][0], score[0], rank, score[1])
        rank += 1

    with open("test.out", mode='w', encoding='UTF-8') as f:
        writer = f
        temp_list = []
        rank = 1
        for score in scores:
            entry = RankingEntry(queries_list[0][0], score[0], rank, score[1])
            temp_list.append(entry)
            rank += 1
        format_run(writer, temp_list, exp_name='test')
        f.close()



    print('end')


def parse_arguments():
    parser = argparse.ArgumentParser(prog='complex-retrieval', description='Complex Answer Retrieval')
    parser.add_argument("outline_file", type=str, help="Location of the outline file", action='store')
    parser.add_argument("paragraph_file", type=str, help="Location of the paragraph file", action='store')

    args = parser.parse_args()
    return args


main()
