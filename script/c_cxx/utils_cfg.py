import numpy as np


def create_adjacency_matrix(save_edge_digit, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])

    for edge in save_edge_digit:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]

        a[tgt_idx][(e_type) * n_nodes + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_nodes + tgt_idx] \
            = 1

    return a


def create_annotation_mat(n_node, annotation_dim, save_node_feature_node_name_int):
    anno = np.zeros([n_node, annotation_dim])
    for i_str, v in save_node_feature_node_name_int.items():
        anno[int(i_str)][v] = 1

    return anno
