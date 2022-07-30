import random 
import re

def get_scans_names_and_embeddings(graph_embeddings_path,
                                scans_rooms_path):

    with open(graph_embeddings_path, 'r') as graph_embeddings_file:
        graph_embeddings_lst = graph_embeddings_file.read().splitlines()
    graph_embeddings_file.close()


    with open(scans_rooms_path, 'r') as scans_rooms_file:
        scans_rooms_lst = scans_rooms_file.read().splitlines()
    scans_rooms_file.close()
    
    return graph_embeddings_lst, scans_rooms_lst


def get_two_random_scans_from_room(scans_rooms_lst, two_random_scans_path):
	for room_scans in scans_rooms_lst:
		room_scans_lst = room_scans.split(" ")
		room_scans_lst.remove('')
		two_random_scans_from_room_lst = random.sample(room_scans_lst, 2)

		with open(two_random_scans_path, 'a') as f:
			for scan in two_random_scans_from_room_lst:
				f.write(scan)
				f.write(' ')
			f.write('\n')
		f.close()



def get_graphs_for_two_scans(two_random_scans_path, graph_embeddings_lst, scans_rooms_lst, database_embeddings_path, test_embeddings_path):
	with open(two_random_scans_path, 'r') as two_random_scans_file:
		random_scans_lst = two_random_scans_file.read().splitlines()
	two_random_scans_file.close()

	for i in range(len(scans_rooms_lst)):
		database_scan_index, test_scan_index = select_two_random_scans(scans_rooms_lst, i, random_scans_lst)
		graph_embedding_for_specific_room_lst = []
		for graph_embedding in graph_embeddings_lst:
			graph_label = re.search(r'\d+', graph_embedding).group()
			if (float(graph_label) == i):
				graph_embedding_for_specific_room_lst.append(graph_embedding)

		five_graph_database_embedding_for_one_scan_lst = graph_embedding_for_specific_room_lst[database_scan_index: database_scan_index+5]
		five_graph_test_embedding_for_one_scan_lst = graph_embedding_for_specific_room_lst[test_scan_index: test_scan_index+5]

		database_embedding_for_one_scan_lst = random.sample(five_graph_database_embedding_for_one_scan_lst, 1)
		test_embedding_for_one_scan_lst = random.sample(five_graph_test_embedding_for_one_scan_lst, 1)

		with open(database_embeddings_path, 'a') as database_embedding_file:
			database_embedding_file.write(''.join(database_embedding_for_one_scan_lst))
			database_embedding_file.write('\n')
		database_embedding_file.close()

		with open(test_embeddings_path, 'a') as test_embedding_file:
			test_embedding_file.write(''.join(test_embedding_for_one_scan_lst))
			test_embedding_file.write('\n')
		test_embedding_file.close()





def select_two_random_scans(scans_rooms_lst, i, random_scans_lst):
	scan_room = scans_rooms_lst[i]
	scan_room_lst = scan_room.split(" ")
	room_scans = random_scans_lst[i]
	two_random_scans_lst = room_scans.split(" ")
	database_scan = two_random_scans_lst[0]
	test_scan = two_random_scans_lst[1]
	database_scan_index = scan_room_lst.index(database_scan)
	test_scan_index = scan_room_lst.index(test_scan)
	return database_scan_index, test_scan_index
		



def main():
    test_scans_rooms_path = 'rooms_cleaned_test.txt'
    graph_embeddings_path = 'test_set.txt'
    graph_embeddings_lst, test_scans_rooms_lst = get_scans_names_and_embeddings(graph_embeddings_path,
                               test_scans_rooms_path)
    
    for i in range(10):
    	two_random_scans_path = f'random_scans/random_scan{i}.txt'             
    	get_two_random_scans_from_room(test_scans_rooms_lst, two_random_scans_path)
    	database_embeddings_path = f'room_embeddings_for_testing/database_embeddings{i}.txt'
    	test_embeddings_path = f'room_embeddings_for_testing/test_embeddings{i}.txt'
    	get_graphs_for_two_scans(two_random_scans_path, graph_embeddings_lst, test_scans_rooms_lst, database_embeddings_path, test_embeddings_path)

                     
main()

