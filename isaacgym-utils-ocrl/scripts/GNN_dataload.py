import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import copy
from tqdm import tqdm

import sys


def adjust_indexing(tuple_list, deleted_index):
    new_tuple_list = []
    for i, j in tuple_list:
        if i > deleted_index:
            i = i-1
        if j > deleted_index:
            j = j-1
        new_tuple_list.append((i,j))
    return new_tuple_list

def remove_duplicate_nodes(edges, init_positions, final_positions, X_force):
    tree_representative = init_positions[0]
    tree_representative = np.around(tree_representative, decimals=4)
    duplicates = [(0,1)] #treat 0 and 1 as duplicates, as 0 represents the base_link aka the floor, which should behave like the root
    for i, node in enumerate(tree_representative):
        for j, nodec in enumerate(tree_representative):
            if (node[:3] == nodec[:3]).all() and i != j and has_same_parent(i,j,edges):
                if i < j:
                    duplicates.append((i,j))
                else:
                    duplicates.append((j,i))
    duplicates = list(set(duplicates))
    while len(duplicates) > 0:
        original, duplicate = duplicates.pop()
        edges, init_positions, final_positions, duplicates, X_force = remove_duplicate(original, duplicate, edges, init_positions, final_positions, duplicates, X_force)
        duplicates = list(set(duplicates))
        duplicates = adjust_indexing(duplicates, duplicate)
        edges = adjust_indexing(edges, duplicate)
    edges = np.array(edges)
    return edges, init_positions[:,:,:3], final_positions[:,:,:3], X_force

def has_same_parent(i,j,edges):
    for parent, child in edges:
        if child == i:
            parent_i = parent
        if child == j:
            parent_j = parent
    return parent_i == parent_j

def remove_duplicate(original, duplicate, edge_def, init_positions, final_positions, duplicates, forces):
    init_positions = np.delete(init_positions, duplicate, axis=1)
    final_positions = np.delete(final_positions, duplicate, axis=1)

    new_edge_def = []
    new_duplicates = []
    for orig, dup in duplicates:
        if orig == duplicate:
            new_duplicates.append((original,dup))
        elif duplicate != dup and duplicate != orig:
            new_duplicates.append((orig,dup))

    for parent, child in edge_def:
        if duplicate == parent:
            new_edge_def.append((original,child))
        elif duplicate != parent and duplicate != child:
            new_edge_def.append((parent,child))

    for idx, force in enumerate(forces):
        if np.linalg.norm(force[duplicate]) != 0:
            forces[idx][original] += forces[idx][duplicate]
    
    forces = np.delete(forces, duplicate, axis=1)

    return new_edge_def, init_positions, final_positions, new_duplicates, forces


def load_npy(data_dir, sim=True):
    if sim:
        # Load npy files from dataset_dir. A shortcut to 'sample_1_push' shared folder has been added to 'My Drive' 
        #X_stiffness_damping = np.load(os.path.join(data_dir, 'X_coeff_stiff_damp.npy'))
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def.npy'))
        X_force = np.load(os.path.join(data_dir, 'final_F.npy'))
        X_pos = np.load(os.path.join(data_dir, 'final_X.npy'))
        Y_pos = np.load(os.path.join(data_dir, 'final_Y.npy'))
 
        # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, n_features)
        X_pos = X_pos[:, :7, :].transpose((0,2,1))
        Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
        X_force = X_force.transpose((0,2,1))

        

    else:
        X_edges = np.load(os.path.join(data_dir, 'X_edge_def.npy'))
        X_force = np.load(os.path.join(data_dir, 'final_F.npy'))
        X_pos = np.load(os.path.join(data_dir, 'final_X.npy'), allow_pickle=True)
        Y_pos = np.load(os.path.join(data_dir, 'final_Y.npy'), allow_pickle=True)

        invalid_graphs = []
        for i, graph in enumerate(X_pos):
            for j, node in enumerate(graph):
                if node is None:
                    invalid_graphs.append(i)

        X_force = np.delete(X_force, invalid_graphs, axis=0)
        X_pos = np.delete(X_pos, invalid_graphs, axis=0)
        Y_pos = np.delete(Y_pos, invalid_graphs, axis=0)
        
        X_pos_list = []
        for graph in X_pos:
            for node in graph:
                for feature in node:
                    X_pos_list.append(feature)
        X_pos = np.array(X_pos_list)
        X_pos = X_pos.reshape(Y_pos.shape[0],Y_pos.shape[1],Y_pos.shape[2]) 
        X_force = X_force.transpose((0,2,1))
    return X_edges, X_force, X_pos, Y_pos

# ===================================================================================
# ==============================  MAIN SCRIPT ======================================
if __name__ == '__main__':

    print(f" ================ starting sample script ================  ")

    params = {
        'run_name': 'mark_withoutDuplicates',
        'dataset_dir': ['/home/mark/data/IROS2023/IROS2023_sim_dataset_1015/10Nodes_by_tree'], #['data/10Nodes_by_tree/trial', 'data/20Nodes_by_tree/trial'],
        'num_trees_per_dir': [10], #[27, 43],
        'simulated_dataset': True,
        'seed': 0,
        'num_epochs': 300, #700
        'batch_size': 512, 
        'lr': 2e-3,
        'train_validation_split': 0.9,
        'remove_duplicates': False
    }


    X_force_list = []
    X_pos_list = []
    Y_pos_list = []
    train_dataset = []
    val_dataset = []

    for i_dir, dataset_dir in enumerate(params['dataset_dir']):
        train_val_split = int(params['num_trees_per_dir'][i_dir]*params['train_validation_split'])

        for i in tqdm(range(0,params['num_trees_per_dir'][i_dir])):
            d = dataset_dir+"/trial" + str(i)
            print(f" directory path: {d} ")
            X_edges, X_force, X_pos, Y_pos = load_npy(d, params['simulated_dataset'])

            # print(f" X_edges.shape: {X_edges.shape} Y_pos.shape: {Y_pos.shape}")

            X_edges, X_force, X_pos, Y_pos = remove_duplicate_nodes(X_edges, X_pos, Y_pos, X_force)

            # print(f" X_edges.shape: {X_edges.shape} Y_pos.shape: {Y_pos.shape}")
            




    print(f" ================ ending script ================  ")
