import numpy as np

from numpy import dot
from numpy.linalg import norm


def get_cos_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def get_embeddings_and_indexes(database_embeddings_path,
                                test_embeddings_path, 
                                database_room_indexes_path, 
                                test_room_indexes_path):

    with open(database_embeddings_path, 'r') as database_embeddings_file:
        database_embeddings_lst = database_embeddings_file.read().splitlines()
    database_embeddings_file.close()


    with open(test_embeddings_path, 'r') as test_embeddings_file:
        test_embeddings_lst = test_embeddings_file.read().splitlines()
    test_embeddings_file.close()

    with open(database_room_indexes_path, 'r') as database_room_indexes_file:
        database_room_indexes_lst = database_room_indexes_file.read().splitlines()
    database_room_indexes_file.close()

    with open(test_room_indexes_path, 'r') as test_room_indexes_file:
        test_room_indexes_lst = test_room_indexes_file.read().splitlines()
    test_room_indexes_file.close()

    return database_embeddings_lst, test_embeddings_lst,database_room_indexes_lst, test_room_indexes_lst


def convert_string_embedding_to_list_embedding(embedding):
    embedding_string_numbers = embedding.split(" ")
    embedding_numbers = [float(x) for x in embedding_string_numbers]
    return embedding_numbers

def check_if_two_room_embeddings_are_equal_top_one(cos_sim_for_top_one_lst, 
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding):
    max_similarity_index = cos_sim_for_top_one_lst.index(max(cos_sim_for_top_one_lst))
    max_similarity_class = database_room_indexes_lst[max_similarity_index]
    test_embedding_index = test_embeddings_lst.index(test_embedding)
    test_embedding_class = test_room_indexes_lst[test_embedding_index]


    is_equal_two_room_embeddings = max_similarity_class == test_embedding_class
    return is_equal_two_room_embeddings

def check_if_two_room_embeddings_are_equal_top_five(cos_sim_for_top_five_lst, 
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding):
    max_similarity_top_five_indexes = (-np.array(cos_sim_for_top_five_lst)).argsort()[:5].tolist()
    max_similarity_top_five_classes = []
    for max_similarity_index in max_similarity_top_five_indexes:
        max_similarity_class = database_room_indexes_lst[max_similarity_index]
        max_similarity_top_five_classes.append(max_similarity_class)

    # print('max_similarity_top_five_classes', max_similarity_top_five_classes)

    test_embedding_index = test_embeddings_lst.index(test_embedding)
    # print('test_embedding_index ', test_embedding_index)
    test_embedding_class = test_room_indexes_lst[test_embedding_index]
    # print('test_embedding_class ', test_embedding_class)
    
    if test_embedding_class in max_similarity_top_five_classes:
        return True
    else: 
        return False




    # print('is_equal_two_room_embeddings_top_five: ', is_equal_two_room_embeddings_top_five)

    if len(is_equal_two_room_embeddings_top_five) > 0:
        return True
    else:
        return False


def get_top_one_score(database_embeddings_path, test_embeddings_path, database_room_indexes_path, test_room_indexes_path):

    database_embeddings_lst, test_embeddings_lst, database_room_indexes_lst, test_room_indexes_lst = get_embeddings_and_indexes(database_embeddings_path,
                                test_embeddings_path, 
                                database_room_indexes_path, 
                                test_room_indexes_path)

    is_equal_two_room_embeddings_lst = []

    for test_embedding in test_embeddings_lst:
        cos_sim_for_top_one_lst = []
        for database_embedding in database_embeddings_lst:
            test_embedding_numbers = convert_string_embedding_to_list_embedding(test_embedding)
            database_embedding_numbers = convert_string_embedding_to_list_embedding(database_embedding)
            cos_sim = get_cos_similarity(test_embedding_numbers, database_embedding_numbers)
            cos_sim_for_top_one_lst.append(cos_sim)

        is_equal_two_room_embeddings = check_if_two_room_embeddings_are_equal_top_one(cos_sim_for_top_one_lst, 
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding)
        is_equal_two_room_embeddings_lst.append(is_equal_two_room_embeddings)

    top_one_accuracy = sum(is_equal_two_room_embeddings_lst)/len(is_equal_two_room_embeddings_lst)
    return top_one_accuracy


def get_top_five_score(database_embeddings_path, test_embeddings_path, database_room_indexes_path, test_room_indexes_path):
    database_embeddings_lst, test_embeddings_lst, database_room_indexes_lst, test_room_indexes_lst = get_embeddings_and_indexes(database_embeddings_path,
                                test_embeddings_path, 
                                database_room_indexes_path, 
                                test_room_indexes_path)

    is_equal_two_room_embeddings_top_five_lst = []

    for test_embedding in test_embeddings_lst:
        cos_sim_for_top_five_lst = []
        for database_embedding in database_embeddings_lst:
            test_embedding_numbers = convert_string_embedding_to_list_embedding(test_embedding)
            database_embedding_numbers = convert_string_embedding_to_list_embedding(database_embedding)
            cos_sim = get_cos_similarity(test_embedding_numbers, database_embedding_numbers)
            cos_sim_for_top_five_lst.append(cos_sim)

        # print('cos_sim_for_top_five_lst: ', cos_sim_for_top_five_lst)

        is_equal_two_room_embeddings_top_five = check_if_two_room_embeddings_are_equal_top_five(cos_sim_for_top_five_lst, 
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding)
        # print('is_equal_two_room_embeddings_top_five ', is_equal_two_room_embeddings_top_five)
        is_equal_two_room_embeddings_top_five_lst.append(is_equal_two_room_embeddings_top_five)
    top_five_accuracy = sum(is_equal_two_room_embeddings_top_five_lst)/len(is_equal_two_room_embeddings_top_five_lst)
    return top_five_accuracy


def main():
    database_embeddings_path = '/home/igor/Desktop/Victoria/tisl/tisl_localization_22s_copy-main/embeddings_and_indeces/database_embeddings.txt'
    test_embeddings_path = '/home/igor/Desktop/Victoria/tisl/tisl_localization_22s_copy-main/embeddings_and_indeces/test_embeddings.txt'
    database_room_indexes_path = '/home/igor/Desktop/Victoria/tisl/tisl_localization_22s_copy-main/embeddings_and_indeces/database_indeces.txt'
    test_room_indexes_path = '/home/igor/Desktop/Victoria/tisl/tisl_localization_22s_copy-main/embeddings_and_indeces/test_indeces.txt'

    top_one_score = get_top_one_score(database_embeddings_path, test_embeddings_path, database_room_indexes_path, test_room_indexes_path)
    top_five_score = get_top_five_score(database_embeddings_path, test_embeddings_path, database_room_indexes_path, test_room_indexes_path)
    
    print('top_one_score: ', top_one_score)
    print('top_five_score: ', top_five_score)

main()

