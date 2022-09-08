import os
import json
from traceback import print_tb
import grakit
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
from os.path import join, isfile
from tkinter.messagebox import NO
from typing import List, Tuple, Dict, Union
import numpy as np
from data import *
import copy


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


def generate_labels(path:str, mode:int) -> Union[dict, Tuple[dict, dict]]:
    '''
    Returns two generated dictionaries for rooms classification 

    Parameters:
        path (str): the path for the text file containng room-scan information,
                    where each line has scan(s) for one room
        mode:
            0 if require no split i.e., all the scans in the given file go to one set;
            1 if require spilt e.g., part of the scans in the given file go to one set, the rest go to another

    Returns:
        if mode == 0:
            labels (Dict[str:int]): key: scan; value: label
        if mode == 1:
            (labels_train, labels_valid) where:
                labels_train (Dict[str:int]): key: scan; value: label;
                labels_valid (Dict[str:int]): key: scan; value: label
    '''
    if mode == 0: 
        # one set
        with open(path, 'r') as f:
            rooms = f.read().splitlines()
        f.close()
        labels = {}
        for i, room in enumerate(rooms):
            scans = room.split()
            for scan in scans:
                labels[scan] = i
        
        return labels

    elif mode == 1: 
        # two splited sets
        with open(path, 'r') as f:
            rooms = f.read().splitlines()
        f.close()
        labels_train = {}
        labels_valid = {}
        for i, room in enumerate(rooms):
            scans = room.split()
            for scan in scans[:-1]:
                labels_train[scan] = i
            labels_valid[scans[-1]] = i
        
        return labels_train, labels_valid

    else:
        print("invalid mode")
        return None


def split_dictionary(raw:dict) -> Tuple[dict, dict]:
    '''
    Split the dictionary into two

    Parameters:
        raw: the given dictionary

    Returns:
        s1: the splited dictionary 1
        s2: the splited dictionary 2
    '''
    pass


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


def rooms_train_test_split(data:str, train_size:int) -> None:
    '''
    Split the given room data into train (actually train & valid) set and test set

    Parameters:
        data: the room data to be splited
        train_size: the size of the train set

    Returns:
        None
    '''
    with open(data, 'r') as f:
        room_lst = f.read().splitlines()
    
    with open('rooms_cleaned_train.txt', 'w') as f:
        f.write('\n'.join(room_lst[:train_size]))

    with open('rooms_cleaned_test.txt', 'w') as f:
        f.write('\n'.join(room_lst[train_size:]))


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


def get_object_class_direct() -> dict:
    '''
    Get object-to-class dictionary

    Parameters:
        threshold: ignore objects with frequency under threshold

    Returns:
        dictionary where key is the object label and the value is its one-hot class
    '''
    obj_lst = OBJ_OF_INTEREST

    # Get objects of interest 
    num_objects_of_interest = len(obj_lst)
    # print(objects_of_interest, num_objects_of_interest)

    # Construct one-hot vector
    a = np.array([x for x in range(num_objects_of_interest)])
    one_hots = np.zeros((a.size, a.max()+1))
    one_hots[np.arange(a.size),a] = 1
    one_hots = torch.from_numpy(one_hots).int()
    # print(one_hots)

    # Associate objects of interest with their one-hot class
    obj_dic = {}
    for i, kind in enumerate(obj_lst):
        for obj in kind:
            obj_dic[obj] = one_hots[i]
    # print(obj_dic["pillow"])
    return num_objects_of_interest, obj_dic


def get_object_info_json(path:str, threshold:int, mask:int) -> Tuple[int, List[Tuple]]:
    '''
    Get object information from .json file

    Parameters:
        path: the .json file path 
        threshold: on the frequency

    Returns:
        number of classes &
        list of information of each object in the .json file
    '''
    if threshold == -1:
        num_of_classes, obj_dic = get_object_class_direct()
    else:
        obj_dic = get_object_class(threshold)
        num_of_classes = len(obj_dic)

    with open(path) as f:
        data = json.load(f)
    f.close()

    # Trick
    if mask == -1:
        mask_lst = []
    else:
        mask_lst = OBJ_OF_INTEREST[mask]

    lst = []
    for obj in data['segGroups']:
        pos = torch.tensor(obj['obb']['centroid'])
        normal = torch.tensor(obj['dominantNormal'])
        raw_label = obj['label']
        if raw_label in obj_dic.keys() and (raw_label not in mask_lst):
            label = obj_dic[raw_label]
            ignore = False
        else:
            label = -1
            ignore = True
        obj_info = (pos, label, ignore)
        lst.append(obj_info)
    
    return num_of_classes, lst


def get_scan_emb_info(scan:str) -> torch.Tensor:
    '''
    Get all the (frame) embeddings for a given scan

    Parameters:
        scan: the name of a scan 

    Returns:
        tch_mtx: a torch tensor contains all the embedding infomation where each row is for one frame:
            row[:3]: the 3D position at where the frame is taken
            row[3:3 + dim(emb)]" the embedding of the frame
    '''
    parent_path = "pca_vlad_embeddings_50_random_images"
    child_path = scan + ".txt"
    path = join(parent_path, child_path)

    np_mtx = np.loadtxt(path, dtype=float)
    tch_mtx = torch.from_numpy(np_mtx)
    
    return tch_mtx


def generate_graphs(data_path:str, f_path:str, scans:dict, num:int, \
    pattern:Tuple[int], pos_ind:int, feat_ind:int, base:int, \
        method:int, ratio:float, threshold:int, mask:int) -> int:
    '''
    Generate graphs for given scans, and write them onto the given .txt file

    Parameters:
        data_path: the name of dataset (for this project is '3rscan')
        f_path: the path of the .txt file we want to write on
        scans: the dictionary[scan,label] contains all the scans, for which we want to generate graphs
        num: number of graphs to generate for each scan
        pattern: which pattern to follow to generate graph
        pos_ind: see @grakit.load_object_vertex
        feat_ind: see @grakit.load_object_vertex
        base: see @grakit.generate_edge
        method: see @grakit.generate_edge
        ratio: see @grakit.generate_edge
        threshold: see @build_dataset_combined

    Returns:
        count: total number of graphs generated
    '''
    count = 0
    for scan, label in scans.items():
        label = str(scans[scan])
        
        # get obj info
        path = join(data_path, scan, 'semseg.v2.json')
        num_class, obj_info = get_object_info_json(path, threshold, mask)

        # then get obj vertex
        obj_v = grakit.load_object_vertex(raw_vs=obj_info, pos_ind=pos_ind, feat_ind=feat_ind)

        # get emb info
        emb_info = get_scan_emb_info(scan)
        if (len(emb_info) < 30) or (len(emb_info) > 50) : continue
        # print(len(emb_info), scan, label)

        # then get emb vertex
        emb_v = grakit.load_emb_vertex(emb_info)

        set = pattern[0]
        part = pattern[1]
        for j in range(num):
            if set == 0: # prototype
                sample_obj_v = random.sample(obj_v, random.randint(0, len(obj_v)))
                sample_emb_v = random.sample(emb_v, random.randint(0, len(emb_v)))
            elif set == 1:
                sample_obj_v = random.sample(obj_v, min(len(obj_v), random.randint(2, 4)))
                sample_emb_v = random.sample(emb_v, min(len(emb_v), random.randint(2, 4)))
            elif set == 2:
                sample_obj_v = random.sample(obj_v, min(len(obj_v), random.randint(6, 8)))
                sample_emb_v = random.sample(emb_v, min(len(emb_v), random.randint(6, 8)))
            elif set == 3:
                sample_obj_v = random.sample(obj_v, min(len(obj_v), random.randint(10, 12)))
                sample_emb_v = random.sample(emb_v, min(len(emb_v), random.randint(10, 12)))
            elif set == 4:
                sample_obj_v = random.sample(obj_v, min(len(obj_v), random.randint(14, 16)))
                sample_emb_v = random.sample(emb_v, min(len(emb_v), random.randint(14, 16)))
            elif set == 5:
                sample_obj_v = random.sample(obj_v, min(len(obj_v), random.randint(18, 20)))
                sample_emb_v = random.sample(emb_v, min(len(emb_v), random.randint(18, 20)))
            elif set == 6:
                sample_obj_v = random.sample(obj_v, min(len(obj_v), random.randint(22, 24)))
                sample_emb_v = random.sample(emb_v, min(len(emb_v), random.randint(22, 24)))
            elif set == 7: # all
                sample_obj_v = obj_v
                sample_emb_v = emb_v
                
            if part == 0: # obj & emb
                sample_v = sample_obj_v + sample_emb_v
            elif part == 1: # No obj
                sample_v = sample_emb_v
            elif part == 2: # No emb
                sample_v = sample_obj_v

            if len(sample_v) == 0: continue
    
            g = grakit.graph(vertices=sample_v, base=base, method=method, ratio=ratio)
            deg_lst = g.get_nbs()[0]
            nbs_lst = g.get_nbs()[1]
        
            f = open(f_path, 'a')
            f.write(str(len(sample_v)) + ' ' + label + '\n')
            for i, v in enumerate(sample_v):
                tag = str(v.tag)
                nbs = " ".join(map(str, nbs_lst[i]))
                feat = " ".join(map(str, v.feat.numpy()))
                if tag == "0": # emb, so pad for obj
                    obj_pad = " ".join(map(str, np.zeros(num_class)))
                    # obj_pad = " ".join(map(str, np.zeros(64)))
                    f.write(" ".join([tag, str(deg_lst[i]), nbs, feat, obj_pad]) + '\n')
                elif tag == "1": # obj, so pad for emb
                    emb_pad = " ".join(map(str, np.zeros(64)))
                    f.write(" ".join([tag, str(deg_lst[i]), nbs, emb_pad, feat]) + '\n')
            f.close()
            count += 1
    return count
    

def build_dataset_train_valid(target_name:str, pos_ind:int, feat_ind:int, base:int, method:int, ratio:float, threshold:int) -> Tuple[int]:
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
        threshold: only consider objects with frequency >= threshold
        mode:
            0 if Train & Valid;
            1 if Train & Test

    Returns:
        None
    '''
    data_path = '3rscan'
    scan_to_label_train, scan_to_label_valid = generate_labels(path='rooms_cleaned_train.txt', mode=1)

    # Initialite the file
    f_path = join('pytorch_DGCNN-master/data', target_name, target_name+'.txt')
    open(f_path, 'w').close()
    
    # Generate graphs for the training set
    count_train = generate_graphs(data_path=data_path, f_path=f_path, scans=scan_to_label_train, num=10, \
        pattern=(0,0), pos_ind=pos_ind, feat_ind=feat_ind, base=base, method=method, ratio=ratio, threshold=threshold, mask=-1)

    # Generate graphs for the validation
    count_valid = generate_graphs(data_path=data_path, f_path=f_path, scans=scan_to_label_valid, num=5, \
        pattern=(0,0), pos_ind=pos_ind, feat_ind=feat_ind, base=base, method=method, ratio=ratio, threshold=threshold, mask=-1)

    # Conclude the file
    with open(f_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(count_train + count_valid) + '\n' + content)
    
    return count_train, count_valid


def build_dataset_test(pattern:Tuple[int], pos_ind:int, feat_ind:int, base:int, method:int, ratio:float, threshold:int, mask:int) -> Tuple[int]:
    '''
    Build test set for DGCNN.

    Parameters:
        rooms_path: the path of the .txt file which stores all rooms
        target_name: the name of the dataset
        pos_ind: see @grakit.load_object_vertex
        feat_ind: see @grakit.load_object_vertex
        base: see @grakit.generate_edge
        method: see @grakit.generate_edge
        ratio: see @grakit.generate_edge
        threshold: only consider objects with frequency >= threshold

    Returns:
        None
    '''
    data_path = '3rscan'
    scan_to_label_test = generate_labels(path='rooms_cleaned_test.txt', mode=0)

    # Initialite the file
    f_path = join('experiment_inputs', '{}.txt'.format(mask))
    # f_path = join('experiment_inputs', '{}_{}.txt'.format(pattern[0], pattern[1]))
    open(f_path, 'w').close()
    
    # Generate graphs for the test set
    count = generate_graphs(data_path=data_path, f_path=f_path, scans=scan_to_label_test, num=5, \
        pattern=pattern ,pos_ind=pos_ind, feat_ind=feat_ind, base=base, method=method, ratio=ratio, threshold=threshold, mask=mask)

    # Conclude the file
    with open(f_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(count) + '\n' + content)
    
    return count


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


def build_train_valid_split_fold(i:int, data:str, count_train:int, count_valid:int) -> None:
    '''
    Build train valid split (1 fold) for a dataset

    Parameters:
        i: in the i-th folders
        data: the name of the dataset 
        count_train: the size of the train split
        count_valid: the size of the valid split

    Returns:
        None
    '''
    ind_list = [x for x in range(count_train + count_valid)]
    test_file = 'test_idx-' + str(i) + '.txt'
    train_file = 'train_idx-' + str(i) + '.txt'
    test_path = join('pytorch_DGCNN-master', 'data', data, '10fold_idx', test_file)
    train_path = join('pytorch_DGCNN-master', 'data', data, '10fold_idx', train_file)
    with open(train_path, 'w') as f:
        for ind in ind_list[:count_train]:
            f.write("{}\n".format(ind))
    f.close()
    with open(test_path, 'w') as f:
        for ind in ind_list[count_train:]:
            f.write("{}\n".format(ind))
    f.close()


def visualize_graph(data:str, node_size:int, nth:int, num_columns:int, figsize:Tuple[int], is_color_graph: bool) -> None:
    '''
    Visualize graphs in the given data

    Parameters:
        data: input .txt file for DGCNN
        node_size: size of the node of graph
        nth: n-th of all graphs to be visualized
        num_columns: number fo columns of the plot
        figsize: figsize of the plot
        is_color_graph: True if you want different classes have different colors

    Returns:
        None
    '''
    f = open(data, 'r')
    # num_classes = 55
    num_graphs = int(f.readline())
    num_rows = 10
    # num_rows = (num_graphs // num_columns) + 1
    # print(num_graphs)
    plt.figure(figsize=figsize)
    for i in range(num_graphs // nth):
        # for one graph
        g_info = f.readline().split()
        num_vertices = int(g_info[0])
        label = g_info[1]
    #     print(num_vertices)
        G = nx.Graph()
        if is_color_graph:
            colors = []
        for j in range(num_vertices):
            # for one vertex
            vertex_info = f.readline().split()
            tag = int(vertex_info[0])
            num_nbs = int(vertex_info[1])
            if is_color_graph:
    #             obj_class = vertex_info[num_nbs+2:].index("1")
                colors.append(tag/2)
            if num_nbs == 0:
                G.add_node(j)
            else:
                for k in range(2, num_nbs + 2):
                    G.add_edge(j, int(vertex_info[k]))
        plt.subplot(num_rows, num_columns, i+1)
        plt.title('label: {}'.format(label))
        options = {
        "font_size": 1,
        "node_size": node_size,
    #     "node_color": "white",
    #     "edgecolors": "black",
    #     "linewidths": 5,
    #     "width": 5,
        }
    #     colors = [i/len(G.nodes) for i in range(len(G.nodes))]
        if is_color_graph:
            nx.draw_networkx(G, **options, node_color=colors)
        else:
            nx.draw_networkx(G, **options)
    f.close()
