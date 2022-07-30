import numpy as np

from numpy import dot
from numpy.linalg import norm


def get_cos_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim
 
def get_euclidian_distance(a, b):
    a_numpy = np.array(a)
    b_numpy = np.array(b)
    euclidian_dist = norm(a_numpy-b_numpy)
    return euclidian_dist 

def get_embeddings_and_indexes(database_embeddings_path,
                                test_embeddings_path):

    with open(database_embeddings_path, 'r') as database_embeddings_file:
        database_embeddings_with_indexes = database_embeddings_file.read().splitlines()
        database_room_indexes_lst = []
        database_embeddings_lst = []
        for database_embeddings_with_index in database_embeddings_with_indexes:
            database_embeddings_with_index_lst = database_embeddings_with_index.split(" ")
            database_room_index = database_embeddings_with_index_lst[0]
            database_room_indexes_lst.append(database_room_index)
            database_embedding = database_embeddings_with_index_lst[1:]
            database_embeddings_lst.append(database_embedding)
    database_embeddings_file.close()


    with open(test_embeddings_path, 'r') as test_embeddings_file:
        test_embeddings_with_indexes = test_embeddings_file.read().splitlines()
        test_room_indexes_lst = []
        test_embeddings_lst = []
        for test_embeddings_with_index in test_embeddings_with_indexes:
            test_embeddings_with_index_lst = test_embeddings_with_index.split(" ")
            test_room_index = test_embeddings_with_index_lst[0]
            test_room_indexes_lst.append(test_room_index)
            test_embedding = test_embeddings_with_index_lst[1:]
            test_embeddings_lst.append(test_embedding)
    test_embeddings_file.close()

    return database_embeddings_lst, test_embeddings_lst, database_room_indexes_lst, test_room_indexes_lst


def convert_string_embedding_to_list_embedding(embedding):
    embedding_numbers = [float(x) for x in embedding]
    return embedding_numbers

def check_if_two_room_embeddings_are_equal_top_one(#cos_sim_for_top_one_lst, 
					      euclidian_dist_for_top_one_lst,
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding):
    #max_similarity_index = cos_sim_for_top_one_lst.index(max(cos_sim_for_top_one_lst))
    #max_similarity_class = database_room_indexes_lst[max_similarity_index]
    min_distance_index = euclidian_dist_for_top_one_lst.index(min(euclidian_dist_for_top_one_lst))
    min_distance_class = database_room_indexes_lst[min_distance_index]
    test_embedding_index = test_embeddings_lst.index(test_embedding)
    test_embedding_class = test_room_indexes_lst[test_embedding_index]


    #is_equal_two_room_embeddings = max_similarity_class == test_embedding_class
    is_equal_two_room_embeddings = min_distance_class == test_embedding_class
    return is_equal_two_room_embeddings

def check_if_two_room_embeddings_are_equal_top_five(#cos_sim_for_top_five_lst, 
						euclidian_dist_for_top_five_lst,
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding):
    #max_similarity_top_five_indexes = (-np.array(cos_sim_for_top_five_lst)).argsort()[:5].tolist()
    #max_similarity_top_five_classes = []
    min_euclidian_dist_top_five_indexes = (-np.array(euclidian_dist_for_top_five_lst)).argsort()[-5:].tolist()
    min_euclidian_dist_top_five_classes = []
    #for max_similarity_index in max_similarity_top_five_indexes:
        #max_similarity_class = database_room_indexes_lst[max_similarity_index]
        #max_similarity_top_five_classes.append(max_similarity_class)
    for min_euclidian_dist_index in min_euclidian_dist_top_five_indexes:
    	min_euclidaian_distance_class = database_room_indexes_lst[min_euclidian_dist_index]
    	min_euclidian_dist_top_five_classes.append(min_euclidaian_distance_class)



    test_embedding_index = test_embeddings_lst.index(test_embedding)

    test_embedding_class = test_room_indexes_lst[test_embedding_index]

    
    #if test_embedding_class in max_similarity_top_five_classes:
        #return True
    if test_embedding_class in min_euclidian_dist_top_five_classes:
    	return True
    else: 
        return False


def get_top_one_score(database_embeddings_path, test_embeddings_path):

    database_embeddings_lst, test_embeddings_lst, database_room_indexes_lst, test_room_indexes_lst = get_embeddings_and_indexes(database_embeddings_path,
                                test_embeddings_path)

    is_equal_two_room_embeddings_lst = []

    for test_embedding in test_embeddings_lst:
        #cos_sim_for_top_one_lst = []
        euclidian_dist_for_top_one_lst = []
        for database_embedding in database_embeddings_lst:
            test_embedding_numbers = convert_string_embedding_to_list_embedding(test_embedding)
            database_embedding_numbers = convert_string_embedding_to_list_embedding(database_embedding)
            #cos_sim = get_cos_similarity(test_embedding_numbers, database_embedding_numbers)
            #cos_sim_for_top_one_lst.append(cos_sim)
            euclidian_dist = get_euclidian_distance(test_embedding_numbers, database_embedding_numbers)
            euclidian_dist_for_top_one_lst.append(euclidian_dist)

        is_equal_two_room_embeddings = check_if_two_room_embeddings_are_equal_top_one(#cos_sim_for_top_one_lst, 
        				     euclidian_dist_for_top_one_lst,
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding)
        is_equal_two_room_embeddings_lst.append(is_equal_two_room_embeddings)

    top_one_accuracy = sum(is_equal_two_room_embeddings_lst)/len(is_equal_two_room_embeddings_lst)
    return top_one_accuracy


def get_top_five_score(database_embeddings_path, test_embeddings_path):
    database_embeddings_lst, test_embeddings_lst, database_room_indexes_lst, test_room_indexes_lst = get_embeddings_and_indexes(database_embeddings_path,
                                test_embeddings_path)

    is_equal_two_room_embeddings_top_five_lst = []

    for test_embedding in test_embeddings_lst:
        #cos_sim_for_top_five_lst = []
        euclidian_dist_for_top_five_lst = []
        for database_embedding in database_embeddings_lst:
            test_embedding_numbers = convert_string_embedding_to_list_embedding(test_embedding)
            database_embedding_numbers = convert_string_embedding_to_list_embedding(database_embedding)
            #cos_sim = get_cos_similarity(test_embedding_numbers, database_embedding_numbers)
            #cos_sim_for_top_five_lst.append(cos_sim)
            euclidian_dist = get_euclidian_distance(test_embedding_numbers, database_embedding_numbers)
            euclidian_dist_for_top_five_lst.append(euclidian_dist)



        is_equal_two_room_embeddings_top_five = check_if_two_room_embeddings_are_equal_top_five(#cos_sim_for_top_five_lst, 
        					euclidian_dist_for_top_five_lst,
                                            database_room_indexes_lst,
                                            test_room_indexes_lst,
                                            test_embeddings_lst,
                                            test_embedding)

        is_equal_two_room_embeddings_top_five_lst.append(is_equal_two_room_embeddings_top_five)
    top_five_accuracy = sum(is_equal_two_room_embeddings_top_five_lst)/len(is_equal_two_room_embeddings_top_five_lst)
    return top_five_accuracy

def compute_average(lst):
    average = sum(lst)/len(lst)
    return average

def main():

    for i in range(10):
        database_embeddings_path = f'room_embeddings_for_testing/database_embeddings{i}.txt'
        test_embeddings_path = f'room_embeddings_for_testing/test_embeddings{i}.txt'
        
        top_one_scores_lst = []
        top_five_scores_lst = []

        top_one_score = get_top_one_score(database_embeddings_path, test_embeddings_path)
        top_five_score = get_top_five_score(database_embeddings_path, test_embeddings_path)

        top_one_scores_lst.append(top_one_score)
        top_five_scores_lst.append(top_five_score)

    average_top_one_score = compute_average(top_one_scores_lst)
    average_top_five_score = compute_average(top_five_scores_lst)
    
    print('average_top_one_score: ', average_top_one_score)
    print('average_top_five_score: ', average_top_five_score)

main()

