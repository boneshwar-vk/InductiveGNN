import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize

def get_mode(v):
    """
    Get the mode of a list or array.
    
    :param v: A list or array of values
    :return: The mode of the list or array
    """
    return pd.Series(v).mode().iloc[0]

def cluster_dist(latent, data, start_idx):
    """
    Calculate the distance between clusters and the pseudotime.
    
    :param latent: Latent representation of the data
    :param data: A dictionary with keys 'cl' for cluster labels
    :param start_idx: The indices of the start cells
    :return: Dictionary with keys 'center', 'clus_dist', and 'dist_order'
    """
    cl = data['cl']
    start_clu = get_mode(cl[start_idx])
    
    latent_df = pd.DataFrame(latent)
    latent_df['cl'] = cl
    total_col_num = latent_df.shape[1] # cols --> cells
    
    # Calculate Euclidean centers for each cluster
    clus_cent = []
    for i in np.unique(latent_df['cl']):
        temp = latent_df[latent_df['cl'] == i].iloc[:, :-1]
        temp_cent = temp.mean().values
        clus_cent.append(np.append(temp_cent, i))
    
    clus_cent = np.array(clus_cent)
    clus_cent_df = pd.DataFrame(clus_cent, columns=list(range(total_col_num)))
    clus_cent_df.index = np.unique(cl)
    
    # Reorder clusters according to start cell id
    clus_cent_df = clus_cent_df.loc[[start_clu] + [x for x in clus_cent_df.index if x != start_clu]]
    
    # Calculate distance matrix for cluster centers
    dist_matrix = pdist(clus_cent_df.iloc[:, :-1], metric='euclidean')
    clus_dist = squareform(dist_matrix)[:, 0]
    
    # Calculate distances from latent pseudotimes
    dist_order = [clus_dist[int(clus_cent_df.loc[clus]['cl'])] for clus in cl]
    
    return {
        'center': clus_cent_df,
        'clus_dist': clus_dist,
        'dist_order': dist_order
    }

def trajectory_inference(data, start_idx, scDHA_res, allCluster, ncores=10, seed=None):
    """
    Perform trajectory inference on single-cell RNA-seq data.
    
    :param data: Dictionary containing gene expression matrix and clustering labels
    :param start_idx: Indexes of the start cells
    :param scDHA_res: Dictionary containing latent and clustering results from scDHA
    :param allCluster: List of clustering results
    :param ncores: Number of processor cores to use (not utilized here)
    :param seed: Seed for reproducibility
    :return: Dictionary with pseudotime, cluster, data cluster centers, milestone network, and graph object
    """
    if seed is not None:
        np.random.seed(seed)
    
    pseudotime = np.zeros(len(data['expr']))
    latent = scDHA_res['latent']
    
    for clus in allCluster:
        data['cl'] = clus
        out = cluster_dist(latent, data, start_idx)
        tmp = np.array(out['dist_order'])
        tmp = tmp / np.max(tmp)
        pseudotime += tmp
    
    cl = scDHA_res['cluster']
    unique_clusters, counts = np.unique(cl, return_counts=True)
    single_cell_clusters = unique_clusters[counts == 1]
    
    # Assign single-cell clusters to the closest group
    for cluster_id in single_cell_clusters:
        temp_index = np.where(cl == cluster_id)[0][0]
        close_cell_index = np.argmin(np.abs(pseudotime[np.arange(len(pseudotime)) != temp_index] - pseudotime[temp_index]))
        if close_cell_index >= temp_index:
            close_cell_index += 1
        cl[temp_index] = cl[close_cell_index]
    
    # Reindex clusters
    for i, cluster_id in enumerate(unique_clusters):
        cl[cl > cluster_id] -= 1
    
    start_clu = get_mode(cl[start_idx])
    
    data_clus_cent = []
    for cl_id in np.unique(cl):
        cl_index = np.where(cl == cl_id)[0]
        clus_pseudotime = np.mean(pseudotime[cl_index])
        tmp_clus_cent = np.mean(latent[cl_index], axis=0)
        data_clus_cent.append(np.append(tmp_clus_cent, clus_pseudotime))
    
    data_clus_cent = np.array(data_clus_cent)
    data_clus_cent_df = pd.DataFrame(data_clus_cent)
    data_clus_cent_df.index = np.unique(cl).astype(str)
    
    dist_matrix = pdist(data_clus_cent[:, :-1], metric='euclidean')
    dist_matrix = squareform(dist_matrix)
    
    # Build Minimum Spanning Tree (MST)
    g = nx.Graph(dist_matrix)
    mst = nx.minimum_spanning_tree(g)
    
    # Add pseudotime as edge weights
    for edge in mst.edges():
        u, v = edge
        mst[u][v]['weight'] = dist_matrix[int(u), int(v)]
    
    # Ensure edge directions based on pseudotime
    edges = list(mst.edges())
    edges = edges[::-1]
    for edge in edges:
        u, v = edge
        if data_clus_cent_df.iloc[int(u), -1] > data_clus_cent_df.iloc[int(v), -1]:
            mst.remove_edge(u, v)
            mst.add_edge(v, u, weight=mst[v][u]['weight'])
    
    edge_lengths = [data_clus_cent_df.iloc[int(edge[0]), -1] - data_clus_cent_df.iloc[int(edge[1]), -1] for edge in edges]
    
    milestone_network = pd.DataFrame({
        'from': [str(edge[0]) for edge in edges],
        'to': [str(edge[1]) for edge in edges],
        'length': edge_lengths,
        'directed': True
    })
    
    return {
        'pseudotime': pseudotime,
        'cluster': cl,
        'data_clus_cent': data_clus_cent_df,
        'milestone_network': milestone_network,
        'g': mst
    }

# Example usage:
# data = {'expr': np.random.rand(100, 10), 'cl': np.random.randint(1, 6, 100)}
# start_idx = np.where(data['cl'] == 1)[0]
# scDHA_res = {'latent': np.random.rand(100, 10), 'cluster': np.random.randint(1, 6, 100)}
# allCluster = [np.random.randint(1, 6, 100) for _ in range(5)]
# result = trajectory_inference(data, start_idx, scDHA_res, allCluster, ncores=2, seed=1)


# # Example usage
# latent = np.random.rand(10, 5)  # Example latent data
# data = pd.DataFrame({'cl': np.random.randint(1, 4, size=10)})  # Example cluster labels
# start_idx = np.array([0, 1, 2])  # Example starting indices

# result = cluster_dist(latent, data, start_idx)
# print(result)
