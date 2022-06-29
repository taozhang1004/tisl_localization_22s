import os
import json
import grakit
import random
from os.path import join, isfile
from tkinter.messagebox import NO
from typing import List, Tuple, Dict
import torch


def get_directories(path:str, print_result=True) -> List[str]:
    '''
    Returns the list of directories within the given path

    Parameters:
        path (str): the given path

    Returns:
        dir_list (List[str]): the list of directories within the given path
    '''
    dir_list = os.listdir(path)
    
    if print_result:
        print("Directories in '", path, "' :")
        print(dir_list)

    return dir_list


def generate_labels(path:str) -> dict:
    '''
    Returns a generated dictionary for rooms classification 

    Parameters:
        path (str): the path for the text file containng room-scan information,
                    where each line has scan(s) for one room

    Returns:
        labels (Dict[str:int]): key: scan; value: label
    '''
    with open(path, 'r') as f:
        rooms = f.read().splitlines()
    f.close()
    labels = {}
    for i, room in enumerate(rooms):
        scans = room.split()
        for scan in scans:
            labels[scan] = i
    return labels


def remove_odd_rooms(rooms:str, excluded:str, rooms_cleaned:str) -> None:
    '''
    Remove rooms, in given 'rooms', which contains any scan from 'excluded'. Then save the new/cleaned set
    of rooms in the same format in 'rooms_cleaned'

    Parameters:
        rooms (str): the path for given rooms
        excluded (str): the path for exluded scans
        rooms_cleaned (str): the path for cleaned rooms

    Returns:
        None
    '''
    with open(excluded, 'r') as excluded_f:
        excluded_lst = excluded_f.read().splitlines()
    excluded_f.close()

    with open(rooms, 'r') as rooms_f:
        rooms_lst = rooms_f.read().splitlines()
    rooms_f.close()
    print('number of rooms originally: {}'.format(len(rooms_lst)))

    for room in rooms_lst:
        scans = room.split()
        for scan in scans:
            if scan in excluded_lst: 
                rooms_lst.remove(room)
                break
                
    print('number of rooms after exclusion: {}'.format(len(rooms_lst)))

    # Test
    fail_lst = []
    checked = True
    for room in rooms_lst:
        scans = room.split()
        for scan in scans:
            if scan in excluded_lst:
                checked = False
                fail_lst.append(scan)
            
    print(len(fail_lst), fail_lst)
    if checked:
        print('Checked :)')
    else:
        print('Fail :(')
        
    with open(rooms_cleaned, 'w') as rooms_cleaned_f:
        rooms_cleaned_f.write('\n'.join(rooms_lst))
    rooms_cleaned_f.close()


def get_object_class(threshold:int) -> dict:
    '''
    Get object-to-class dictionary

    Parameters:
        threshold: ignore objects with frequency under threshold

    Returns:
        dictionary where key is the object label and the value is its one-hot class
    '''
    # Get objects of interest 
    objects = []
    objects_of_interest = []
    with open('objects_and_frequency.txt', 'r') as f:
        objects = f.readlines()
    for object in objects:
        freq = int(object.split(",")[1])
        obj = object.split(",")[0]
        if freq > threshold:
            objects_of_interest.append(obj)
    num_objects_of_interest = len(objects_of_interest)
    # print(objects_of_interest, num_objects_of_interest)

    # Construct one-hot vector
    import numpy as np
    a = np.array([x for x in range(num_objects_of_interest)])
    one_hots = np.zeros((a.size, a.max()+1))
    one_hots[np.arange(a.size),a] = 1
    one_hots = torch.from_numpy(one_hots).int()
    # print(one_hots)

    # Associate objects of interest with their one-hot class
    obj_dic = {}
    for i in range(num_objects_of_interest):
        obj_dic[objects_of_interest[i]] = one_hots[i]
    # print(obj_dic["pillow"])
    return obj_dic


def get_object_info_json(path:str) -> List[Tuple]:
    '''
    Get object information from .json file

    Parameters:
        path: the .json file path 

    Returns:
        list of information of each object in the .json file
    '''
    # Change threshold
    obj_dic = get_object_class(threshold=100)

    with open(path) as f:
        data = json.load(f)
    f.close()

    lst = []
    for obj in data['segGroups']:
        pos = torch.tensor(obj['obb']['centroid'])
        normal = torch.tensor(obj['dominantNormal'])
        raw_label = obj['label']
        if raw_label in obj_dic.keys():
            label = obj_dic[raw_label]
            ignore = False
        else:
            label = -1
            ignore = True
        obj_info = (pos, label, normal, ignore)
        lst.append(obj_info)
    
    return lst


def build_dataset(rooms_path:str, target_name:str, pos_ind:int, feat_ind:int, base:int, method:int, ratio:float) -> None:
    '''
    Build dataset for DGCNN.

    Parameters:
        rooms_path: the path of the .txt file which stores all rooms
        target_name: the name of the dataset
        pos_ind: see @grakit.load_object_vertex
        feat_ind: see @grakit.load_object_vertex
        base: see @grakit.generate_edge
        method: see @grakit.generate_edge
        ratio: see @grakit.generate_edge

    Returns:
        None
    '''
    data_path = '3rscan'
    scan_to_label = generate_labels(rooms_path)

    f_path = join('pytorch_DGCNN-master/data', target_name, target_name+'.txt')
    f = open(f_path, 'w')
    f.write(str(len(scan_to_label))+'\n')
    f.close()
    for scan, label in scan_to_label.items():
        label = str(scan_to_label[scan])
        
        # get obj info
        path = join(data_path, scan, 'semseg.v2.json')
        obj_info = get_object_info_json(path)

        # then get obj vertex
        obj_v = grakit.load_object_vertex(raw_vs=obj_info, pos_ind=pos_ind, feat_ind=feat_ind)
        obj_g = grakit.graph(vertices=obj_v, base=base, method=method, ratio=ratio)
        deg_lst = obj_g.get_nbs()[0]
        nbs_lst = obj_g.get_nbs()[1]
        
        f = open(f_path, 'a')
        f.write(str(len(obj_v)) + ' ' + label + '\n')
        tag = '0'
        for i, v in enumerate(obj_v):
            nbs = " ".join(map(str, nbs_lst[i]))
            feat = " ".join(map(str, v.feat.numpy()))
            f.write(" ".join([tag, str(deg_lst[i]), nbs, feat]) + '\n')
        f.close()


def build_cv_split_10fold(data:str, total_size:int, test_size:int) -> None:
    '''
    Build cross-validation splite (10 fold) for a dataset

    Parameters:
        data: the name of the dataset 
        total_size: the size of the dataset
        test_size: the size of the test split

    Returns:
        None
    '''
    train_size = total_size - test_size
    ind_list = [x for x in range(total_size)]
    random.shuffle(ind_list)
    for i in range(1,11):
        test_file = 'test_idx-' + str(i) + '.txt'
        train_file = 'train_idx-' + str(i) + '.txt'
        test_path = join('pytorch_DGCNN-master', 'data', data, '10fold_idx', test_file)
        train_path = join('pytorch_DGCNN-master', 'data', data, '10fold_idx', train_file)
        random.shuffle(ind_list)
        with open(train_path, 'w') as f:
            for ind in ind_list[:train_size]:
                f.write("{}\n".format(ind))
        f.close()
        with open(test_path, 'w') as f:
            for ind in ind_list[train_size:]:
                f.write("{}\n".format(ind))
        f.close()