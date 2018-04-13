from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from text_preprocess import Preprocessing
from trec_car.format_runs import *


def save_scores_to_file(output_entries, filename="test.out"):
    with open(filename, mode='w', encoding='UTF-8') as f:
        writer = f
        temp_list = []
        for entry in output_entries:
            temp_list.append(entry)
        format_run(writer, temp_list, exp_name='test')
        f.close()


def create_save_model(sentences, total_words, save=True):
    model = Doc2Vec(dm=0, dbow_words=1, alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    model.build_vocab(sentences)

    for epoch in range(10):
        print("train")
        model.train(sentences, total_examples=len(sentences), total_words=total_words, epochs=10, word_count=1)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    if save:
        model.save('doc2vec_models/test200.v2.0_all.test200.cbor.paragraphs_3.doc2vec')
    return model


preprocessing = Preprocessing("test200.v2.0\\all.test200.cbor.outlines", "test200.v2.0\\all.test200.cbor.paragraphs")
queries_list = preprocessing.get_raw_queries(qe_synonyms=False)
paragraphs_dict = preprocessing.get_raw_paragraphs()

paragraph_id_mapping = {}
paragraph_id_mapping_index = 0
sentences = []
total_words = 0
for p_id, p_text in paragraphs_dict.items():
    paragraph_id_mapping[paragraph_id_mapping_index] = p_id
    sentences.append(TaggedDocument(words=p_text, tags=[paragraph_id_mapping_index]))
    total_words += len(p_text)
    paragraph_id_mapping_index += 1

# model = create_save_model(sentences, total_words, save=False)
model = Doc2Vec.load('doc2vec_models/test200.v2.0_all.test200.cbor.paragraphs_3.doc2vec')

# query_sentence = queries_list[10][2]
# most_similar = model.docvecs.most_similar(positive=[model.infer_vector(query_sentence)], topn=5)

# print(query_sentence)
# i = 1
# for similar in most_similar:
#     par_id = similar[0]
#     similarity = similar[1]
#     original_par_id = paragraph_id_mapping[par_id]
#
#     print(i, similarity, original_par_id, paragraphs_dict[original_par_id])
#     i += 1


output_entries = []
for query in queries_list:
    rank = 1
    query_id = query[0]
    query_sentence = query[2]
    score = model.docvecs.most_similar(positive=[model.infer_vector(query_sentence)], topn=10)
    # score = model.most_similar_cosmul(positive=[model.infer_vector(query_sentence)], topn=10)

    for paragraph_score in score:
        par_id = paragraph_score[0]
        original_par_id = paragraph_id_mapping[par_id]
        par_score = paragraph_score[1]

        entry = RankingEntry(query_id, original_par_id, rank, par_score)
        output_entries.append(entry)
        rank += 1

save_scores_to_file(output_entries, "doc2vec.out")
