from typing import List, Tuple, Union, Any
from scipy.spatial import distance
import torch
import numpy as np


class vertex(object):
    """
    A class to represent a vertex.

    ...

    Attributes
    ----------
    tag : int
        tag of the vertex. 
        e.g. 0 indicates a frame, 1 indicates an object, 2 indicates a wall
    pos : torch.Tensor
        3D position
    feat : ANY
        feature

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, tag:int, pos:torch.Tensor, feat: Any) -> None:
        self.tag = tag # int
        self.pos = pos # torch.Tensor
        self.feat = feat # ANY

    def __str__(self) -> str:
        return 'vertex | tag: {}\n\
            pos: {}\n\
                feat: {}'.format(self.tag, self.pos, self.feat)


class obj_vertex(vertex):
    pass


def generate_edge(vertices:List[vertex], base:int, method:int, ratio:float) -> np.ndarray:
    '''
    Given a set of vertices, generate edge between them, using one edge policy.

    Parameters:
        vertices: list of vertices
        base: indicates which attribute of the vertex the policy is based on
            0 if based on 3D position, 
            1 if based on feature
        method: indicates which method to decide whether there is an egde
            0 if using l2 distance
            1 if using similarity test
        ratio: ratio for method

    Returns:
        edge matrix: classic adjacency matrix but with diagonal being all 0
    '''
    if method == 0: # l2 distance
        num_v = len(vertices)

        if base == 0: # based on pos
            lst = [v.pos for v in vertices]
        else: # based on feat
            # for v in vertices: print(v.feat.shape)
            lst = [v.feat for v in vertices]
        _matrix = torch.stack(lst)
        # print('_matrix.shape:', _matrix.shape)

        # shape: (num_v, num_v)
        dis_matrix = distance.cdist(_matrix, _matrix, 'euclidean')
        average_dis = np.sum(dis_matrix) / (num_v*(num_v - 1))
        threshold = average_dis * ratio
        
        # shape: (num_v, num_v)
        edge_matrix_raw = dis_matrix <= threshold
        f_diag = np.ones((num_v, num_v))
        np.fill_diagonal(f_diag, 0)
        
        edge_matrix = np.logical_and(edge_matrix_raw, f_diag)
        #print('edge_matrix.shape ', edge_matrix.shape)
        return edge_matrix
    
    elif method == 1: # similarity test
        num_v = len(vertices)
        if base == 0: # based on pos
            lst = [v.pos for v in vertices]
        else: # based on feat
            # for v in vertices: print(v.feat.shape)
            lst = [v.feat for v in vertices]
        _matrix = torch.stack(lst)

        dis_matrix = distance.cdist(_matrix, _matrix, 'euclidean')
        n_neighbors = 3
        if dis_matrix.shape[0]<n_neighbors + 1: 
            f_diag = np.ones((num_v, num_v))
            np.fill_diagonal(f_diag, 0)
            #print('f_diag.shape ', f_diag.shape)
            return f_diag
        else:
            f_diag = np.ones((num_v, num_v))
            np.put_along_axis(dis_matrix,np.argpartition(dis_matrix,n_neighbors,axis=1)[:,n_neighbors:],0,axis=1)
            edge_matrix_raw = dis_matrix > 0
            f_diag = np.ones((num_v, num_v))
            np.fill_diagonal(f_diag, 0)
            edge_matrix = np.logical_and(edge_matrix_raw, f_diag)
            #print('edge_matrix.shape ', edge_matrix.shape)
            return edge_matrix
    else:
        print('invalid argument')
        return None


class graph(object):
    """
    A class to represent a graph (unweighted & indirected). 

    ...

    Attributes
    ----------
    V : List[vertex]
        list of integers indicating the indices of the neighbors of the vertex
    V_size : int
        number of vertices in the graph
    E : ndarray
        edge matrix, which is the classic adjacency matrix but with diagonal being all 0

    Methods
    -------
    get_nbs():
        Return the degree list and neighbor list of vertices
    """
    def __init__(self, vertices:List[vertex], base:int, method:int, ratio:float) -> None:
        self.V = vertices
        self.V_size = len(vertices)
        self.E = generate_edge(vertices, base, method, ratio)

    def get_nbs(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        '''
        Returns (deg_lst, nbs_lst_lst), where 
        1) deg_lst == [deg_i] where deg_i is the degree of the i-th vertex; 
        2) nbs_lst_lst == [nbs_lst_i] where nbs_lst_i is the list of indices of neighbors of the i-th vertex

        Parameters:
            self

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: degree list and neighbor list
        '''
        deg_lst = np.sum(self.E, axis=1)
        nbs_lst_lst = [np.where(self.E[i])[0] for i in range(self.V_size)]
        return (deg_lst, nbs_lst_lst)


def load_object_vertex(raw_vs:List[Tuple], pos_ind:int, feat_ind:int) -> List[vertex]:
    '''
    Given a list of tuple representing raw data and return the list of vertices from of it

    Parameters:
        raw_vs : the raw data for object information
        pos_ind: position index, indicating which index of the object information to be the position attribute of the vertex
        feat_ind: feature index, indicating which index of the object information to be the feature attribute of the vertex
        
    Returns:
        List[vertex]: The list of vertex representation
    '''
    lst = []
    for raw_v in raw_vs:
        if not raw_v[-1]:
            vertex_v = vertex(1, raw_v[pos_ind], raw_v[feat_ind])
            lst.append(vertex_v)
    return lst


def load_wall_vertex(raw_vs:List[Tuple], pos_ind:int, feat_ind:int) -> List[vertex]:
    '''
    Given a list of tuple representing raw data and return the list of vertices from of it

    Parameters:
        raw_vs : the raw data for object information
        pos_ind: position index, indicating which index of the object information to be the position attribute of the vertex
        feat_ind: feature index, indicating which index of the object information to be the feature attribute of the vertex
        
    Returns:
        List[vertex]: The list of vertex representation
    '''
    lst = []
    for raw_v in raw_vs:
        if not raw_v[-1]:
            vertex_v = vertex(1, raw_v[pos_ind], raw_v[feat_ind])
            lst.append(vertex_v)
    return lst


def load_emb_vertex(emb_info:torch.Tensor) -> List[vertex]:
    '''
    Given a 2D tensor representing raw data and return the list of vertices from of it

    Parameters:
        emb_info : 2D tensor
        
    Returns:
        List[vertex]: The list of vertex representation
    '''
    lst = []
    for row in emb_info:
        vertex_v = vertex(0, row[:3], row[3:])
        lst.append(vertex_v)
    return lst
