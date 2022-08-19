import random
from typing import List, Tuple, Union, Any

def get_sample_rooms_lst(sample_rooms_path:str) -> List[str]:
    '''
    Given a path to rooms with scans return a list 
    of rooms with corresponding scans

    Parameters:
        sample_rooms_path : path to the file with scans for each room
    
    Returns:
        List[str]: The list for scans for each room
    
    '''
    with open(sample_rooms_path, 'r') as sample_rooms_file:
        sample_rooms_lst = sample_rooms_file.read().splitlines()
    sample_rooms_file.close()
    return sample_rooms_lst

def get_vlad_embedding_lst(scan:str, vlad_embeddings_path:str) -> List[str]:
    '''
    Given a path to vlad embeddings list return
    vlad embeddings to specific scan

    Parameters:
        scan : scan identifier
        vlad_embeddings_path: path to all vlad embeddings
    
    Returns:
        List[str]: The list for vlad embeddings for scan
    
    '''
    vlad_embedding_path_specific_path = vlad_embeddings_path + '/' + scan + '.txt'
    with open(vlad_embedding_path_specific_path, 'r') as vlad_embedding_file:
        vlad_embedding_lst = vlad_embedding_file.read().splitlines()
    vlad_embedding_file.close()
    return vlad_embedding_lst

def get_netvlad_vectors_for_rooms(sample_rooms_lst:List[str], vlad_embeddings_with_room_class_path:str, vlad_embeddings_path:str,  num_emb_for_scan:int) -> None:
    '''
    Given a list of scans to corresponding rooms, path to vlad embeddings with room class, path to vlad embeddings and
    number of embeddings for each scan, write vlad embeddings to the file with class of the room

    Parameters:
        sample_rooms_lst : list of scans to corresponding rooms
        vlad_embeddings_with_room_class_path : vlad_embeddings_with_room_class_path
        vlad_embeddings_path : path to vlad embeddings
        num_emb_for_scan : number of embeddings for each scan

    Returns:
        None

    '''

    vlad_embeddings_with_room_class_path_n_emb = vlad_embeddings_with_room_class_path + f'test_set_vlad_embeddings_{num_emb_for_scan}_emb.txt'
    count = 0 
    for scans in sample_rooms_lst:

        scans_lst = scans.split(" ")
        scans_lst = scans_lst[:-1]
        
        for scan in scans_lst:
            vlad_embedding_lst = get_vlad_embedding_lst(scan, vlad_embeddings_path)
            random_vlad_embedding_lst = random.sample(vlad_embedding_lst, min(num_emb_for_scan, len(vlad_embedding_lst)))
            with open(vlad_embeddings_with_room_class_path_n_emb, 'a') as vlad_embedding_with_room_class_file:
                for vlad_embedding in random_vlad_embedding_lst:
                    vlad_embedding_without_coordinates = vlad_embedding.split(" ")[3:]
                    vlad_embedding_with_room_class_file.write(str(count))
                    vlad_embedding_with_room_class_file.write(' ')
                    vlad_embedding_with_room_class_file.write(' '.join(vlad_embedding_without_coordinates))
                    vlad_embedding_with_room_class_file.write('\n')
            vlad_embedding_with_room_class_file.close()
        count +=1 

def main():
    sample_rooms_path = 'rooms_cleaned_test.txt'
    sample_rooms_lst = get_sample_rooms_lst(sample_rooms_path)

    vlad_embeddings_with_room_class_path = 'test_sets/'
    vlad_embeddings_path = 'pca_vlad_embeddings_50_random_images'

    num_emb_for_scan_lst = [5, 25, 50]
    for num_emb_for_scan in num_emb_for_scan_lst:
        get_netvlad_vectors_for_rooms(sample_rooms_lst, vlad_embeddings_with_room_class_path, vlad_embeddings_path,  num_emb_for_scan)

main()