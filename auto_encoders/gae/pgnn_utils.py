import multiprocessing as mp
import random
from multiprocessing import get_context
import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm

def get_communities(adj_matrix, features):
    graph = nx.from_numpy_matrix(adj_matrix)
   # Randomly rewire 1% edges
    node_list = list(graph.nodes)
    for u, v in graph.edges():
        if random.random() < 0.005:
            x = random.choice(node_list)
            if graph.has_edge(u, x):
                continue
            graph.remove_edge(u, v)
            graph.add_edge(u, x)

    # remove self-loops
    graph.remove_edges_from(nx.selfloop_edges(graph))
    edge_index = np.array(list(graph.edges))
    # Add (i, j) for an edge (j, i)
    edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
    edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

    # if remove_feature:
    #     feature = torch.ones((n, 1))
    # else:
    #     rand_order = np.random.permutation(n)
    #     feature = np.identity(n)[:, rand_order]

    data = {
        "edge_index": edge_index,
        "feature": features,
        "num_nodes": features.shape[0],
    }
    return data


def to_single_directed(edges):
    edges_new = np.zeros((2, edges.shape[1] // 2), dtype=int)
    j = 0
    for i in range(edges.shape[1]):
        if edges[0, i] < edges[1, i]:
            edges_new[:, j] = edges[:, i]
            j += 1

    return edges_new


# each node at least remain in the new graph
def split_edges(data, non_train_ratio=0.2):
    edges = to_single_directed(data["edge_index"].numpy())
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    split1 = int((1 - non_train_ratio) * e)
    split2 = int((1 - non_train_ratio / 2) * e)

    data.update(
        {
            "edges_train": edges[:, :split1],  # 80%
            "edges_val": edges[:, split1:split2],  # 10%
            "edges_test": edges[:, split2:],  # 10%
        }
    )
    return data

def to_bidirected(edges):
    return np.concatenate((edges, edges[::-1, :]), axis=-1)


def shortest_path(graph, node_range, cutoff):
    dists_dict = {}
    for node in tqdm(node_range, leave=False):
        dists_dict[node] = nx.single_source_shortest_path_length(
            graph, node, cutoff
        )
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path(graph, cutoff=None, num_workers=2):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    pool = mp.Pool(processes=num_workers)
    interval_size = len(nodes) / num_workers
    results = [
        pool.apply_async(
            shortest_path,
            args=(
                graph,
                nodes[int(interval_size * i) : int(interval_size * (i + 1))],
                cutoff,
            ),
        )
        for i in range(num_workers)
    ]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
    """
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    """
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes # 124 for goolam et al (num cells)
    dists_array = np.zeros((n, n))
    dists_dict = all_pairs_shortest_path(
        graph, cutoff=approximate if approximate > 0 else None
    )
    node_list = graph.nodes()
    for node_i in node_list:
        shortest_dist = dists_dict[node_i]
        for node_j in node_list:
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array


def get_dataset(args,adj_matrix, features):
    # Generate graph data
    data = get_communities(adj_matrix, features)
    # Get positive and negative edges
    data = split_edges(data=data,non_train_ratio=0.2)

    dists = precompute_dist_data(
        data["edge_index"].numpy(),
        data["num_nodes"],
        approximate=args.k_hop_dist,
    )
    data["dists"] = torch.from_numpy(dists).float() # data dist shape --> nxn (124,124) 

    return data


def get_anchors(n):
    """Get a list of NumPy arrays, each of them is an anchor node set"""
    m = int(np.log2(n))
    anchor_set_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for _ in range(m):
            anchor_set_id.append(
                np.random.choice(n, size=anchor_size, replace=False)
            )
    return anchor_set_id

def get_dist_max(anchor_set_id, dist):
    # N x K, N is number of nodes, K is the number of anchor sets
    dist_max = torch.zeros((dist.shape[0], len(anchor_set_id)))
    dist_argmax = torch.zeros((dist.shape[0], len(anchor_set_id))).long()

    for i in range(len(anchor_set_id)):
        temp_id = torch.as_tensor(anchor_set_id[i], dtype=torch.long)
        # Get reciprocal of shortest distance to each node in the i-th anchor set
        dist_temp = torch.index_select(dist, 1, temp_id)
        # For each node in the graph, find its closest anchor node in the set
        # and the reciprocal of shortest distance
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = torch.index_select(temp_id, 0, dist_argmax_temp)
    return dist_max, dist_argmax


def get_a_graph(dists_max, dists_argmax):
    src = []
    dst = []
    real_src = []
    real_dst = []
    edge_weight = []
    dists_max = dists_max.numpy()
    for i in range(dists_max.shape[0]):
        # Get unique closest anchor nodes for node i across all anchor sets
        tmp_dists_argmax, tmp_dists_argmax_idx = np.unique(
            dists_argmax[i, :], True
        )
        src.extend([i] * tmp_dists_argmax.shape[0])
        real_src.extend([i] * dists_argmax[i, :].shape[0])
        real_dst.extend(list(dists_argmax[i, :].numpy()))
        dst.extend(list(tmp_dists_argmax))
        edge_weight.extend(dists_max[i, tmp_dists_argmax_idx].tolist())
    eid_dict = {(u, v): i for i, (u, v) in enumerate(list(zip(dst, src)))}
    anchor_eid = [eid_dict.get((u, v)) for u, v in zip(real_dst, real_src)]
    g = (dst, src)
    return g, anchor_eid, edge_weight


def get_graphs(data, anchor_sets):
    graphs = []
    anchor_eids = []
    dists_max_list = []
    edge_weights = []
    for anchor_set in tqdm(anchor_sets, leave=False):
        dists_max, dists_argmax = get_dist_max(anchor_set, data["dists"])
        g, anchor_eid, edge_weight = get_a_graph(dists_max, dists_argmax)
        graphs.append(g)
        anchor_eids.append(anchor_eid)
        dists_max_list.append(dists_max)
        edge_weights.append(edge_weight)

    return graphs, anchor_eids, dists_max_list, edge_weights


def merge_result(outputs):
    graphs = []
    anchor_eids = []
    dists_max_list = []
    edge_weights = []

    for g, anchor_eid, dists_max, edge_weight in outputs:
        graphs.extend(g)
        anchor_eids.extend(anchor_eid)
        dists_max_list.extend(dists_max)
        edge_weights.extend(edge_weight)
    return graphs, anchor_eids, dists_max_list, edge_weights

# changed workers form 4 to 1
def preselect_anchor(data, args, num_workers=4):
    pool = get_context("spawn").Pool(processes=num_workers)
    # Pre-compute anchor sets, a collection of anchor sets per epoch
    anchor_set_ids = [
        get_anchors(data["num_nodes"]) for _ in range(args.graph_AE_epoch)
    ] # 200 for 200 epochs
    interval_size = len(anchor_set_ids) / num_workers
    results = [
        pool.apply_async(
            get_graphs,
            args=(
                data,
                anchor_set_ids[
                    int(interval_size * i) : int(interval_size * (i + 1))
                ],
            ),
        )
        for i in range(num_workers)
    ]

    output = [p.get() for p in results]
    graphs, anchor_eids, dists_max_list, edge_weights = merge_result(output)
    pool.close()
    pool.join()
    print("shape of dists_max_lists ------->m ", len(dists_max_list), dists_max_list[0].shape)

    return graphs, anchor_eids, dists_max_list, edge_weights

