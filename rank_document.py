import argparse

from text_preprocess import Preprocessing


def main():
    args = parse_arguments()
    preprocessing = Preprocessing(args.outline_file, args.paragraph_file)
    queries_dict = preprocessing.get_queries()
    paragraphs_dict = preprocessing.get_paragraphs()
    print('end')


def parse_arguments():
    parser = argparse.ArgumentParser(prog='complex-retrieval', description='Complex Answer Retrieval')
    parser.add_argument("outline_file", type=str, help="Location of the outline file", action='store')
    parser.add_argument("paragraph_file", type=str, help="Location of the paragraph file", action='store')

    args = parser.parse_args()
    return args

main()