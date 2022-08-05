import random

def get_sample_rooms_lst(sample_rooms_path):
    with open(sample_rooms_path, 'r') as sample_rooms_file:
        sample_rooms_lst = sample_rooms_file.read().splitlines()
    sample_rooms_file.close()
    return sample_rooms_lst

def main():

    sample_rooms_path = 'rooms_cleaned_test.txt'
    sample_rooms_lst = get_sample_rooms_lst(sample_rooms_path)

    vlad_embeddings_with_room_class_path = 'test_set_vlad_embeddings.txt'
    vlad_embeddings_path = '/home/igor/Desktop/Victoria/tisl/tisl_localization_22s_copy-main/pca_vlad_embeddings_50_random_images'


    count = 0 
    for scans in sample_rooms_lst:

        scans_lst = scans.split(" ")
        scans_lst = scans_lst[:-1]
        
     
        for scan in scans_lst:
            vlad_embedding_path_specific_path = vlad_embeddings_path + '/' + scan + '.txt'
            with open(vlad_embedding_path_specific_path, 'r') as vlad_embedding_file:
                vlad_embedding_lst = vlad_embedding_file.read().splitlines()
            vlad_embedding_file.close()

            random_vlad_embedding_lst = random.sample(vlad_embedding_lst, 5)
            with open(vlad_embeddings_with_room_class_path, 'a') as vlad_embedding_with_room_class_file:
                for vlad_embedding in random_vlad_embedding_lst:
                    vlad_embedding_without_coordinates = vlad_embedding.split(" ")[3:]
                    vlad_embedding_with_room_class_file.write(str(count))
                    vlad_embedding_with_room_class_file.write(' ')
                    vlad_embedding_with_room_class_file.write(' '.join(vlad_embedding_without_coordinates))
                    vlad_embedding_with_room_class_file.write('\n')
            vlad_embedding_with_room_class_file.close()
        count +=1 
            

    










main()





