import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import EoN
from collections import defaultdict
import random
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import pymetis
import math

def add_noise_notestate(df_list, client_ratio, node_ratio, data_frac):
    # 确定要添加噪声的子图数量
    K = len(df_list)
    num_selected_clients = int(math.ceil(client_ratio * K))
    #selected_clients = random.sample(range(K), num_selected_clients)
    selected_clients = list(range(K))[:num_selected_clients]
    # 生成新的 df_list，保留原始数据不变
    df_list_noise = df_list.copy()
    
    # 对每个被选中的 df_subgraph 添加噪声
    for idx in selected_clients:
        df_subgraph = df_list_noise[idx].copy()
        
        # 计算要修改的时间步和节点数量
        T = df_subgraph.shape[0]
        N_k = df_subgraph.shape[1]
        max_time_steps = int(T * data_frac)
        num_noisy_cells = int(node_ratio * N_k)
        
        # 在前 max_time_steps 时间步内添加噪声
        for t in range(max_time_steps):
            # 获取该时间步中状态为 1 的所有节点
            state_1_indices = df_subgraph.iloc[t][df_subgraph.iloc[t] == 1].index.tolist()
            
            # 随机选择 num_noisy_cells 个节点，将状态从 1 修改为 0
            if len(state_1_indices) >= num_noisy_cells:
                selected_indices = random.sample(state_1_indices, num_noisy_cells)
                df_subgraph.iloc[t][selected_indices] = 0
            else:
                selected_indices = random.sample(state_1_indices,len(state_1_indices))
                df_subgraph.iloc[t][selected_indices] = 0
        
        # 更新 df_list_noise 中的子图
        df_list_noise[idx] = df_subgraph

    return df_list_noise

# generate epidemic data on airport network
def construct_traffic_network():
    edges = pd.read_csv('data/edges.csv', sep=',')
    nodes = pd.read_csv('data/nodes.csv', sep=',')

    G = nx.DiGraph()
    G.add_nodes_from(nodes['# index'])

    for index, edge in edges.iterrows():
        if G.has_edge(edge['# source'], edge[' target']):
            G[edge['# source']][edge[' target']]['weight'] += 1
        else:
            G.add_edge(edge['# source'], edge[' target'], weight=1)

    print("===Airport traffic network constructed===", G.number_of_nodes(), G.number_of_edges())
    return G

def base_network(G, num):
    if num > 0:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num]
        top_node_ids = [node[0] for node in top_nodes]
        subG = G.subgraph(top_node_ids).copy()

    else:
        subG = G.copy()
    
    subG.remove_nodes_from(list(nx.isolates(subG)))    
    # print(list(subG.nodes()))
    print("===Airport traffic network constructed===", subG.number_of_nodes(), subG.number_of_edges()) 
    
    # 创建一个新的无权无向图
    undirected_subG = nx.Graph()

    # 遍历原始图中的所有边并添加到新图中
    for u, v, data in subG.edges(data=True):
        if data['weight'] > 0:
            undirected_subG.add_edge(u, v)
    
    degrees = [undirected_subG.degree(n) for n in undirected_subG.nodes()]
    plt.hist(degrees)
    plt.show()

    return undirected_subG

def separate_graph_K_clients(G, num_client, method):
    subgraphs = []
    N = G.number_of_nodes()
    K = num_client
    
    if method == 'evenly_sequence':
            # 将节点映射为整数索引
        nodes = list(G.nodes())    
        # print(nodes)
        # 计算每个簇的节点数
        nodes_per_cluster = N // K        
        # 创建子图列表
        subgraphs = []
        for i in range(K):
            start_index = i * nodes_per_cluster
            end_index = (i + 1) * nodes_per_cluster
            nodes_in_cluster = nodes[start_index:end_index]
            subgraph = G.subgraph(nodes_in_cluster).copy()
            subgraphs.append(subgraph)
        
    elif method == 'spectral_clustering':
        L = nx.normalized_laplacian_matrix(G).toarray()
        n_clusters = num_client
        spectral_clustering = SpectralClustering(
            n_clusters=num_client, 
            affinity='precomputed',  # 仍然使用邻接矩阵，但也可以使用 'rbf' 等
            assign_labels='kmeans',  # 使用 kmeans 进行最终聚类
            random_state=0
        )
        adj_matrix = nx.to_numpy_array(G)
        clusters = spectral_clustering.fit_predict(adj_matrix)
        subgraphs = []
        for i in range(n_clusters):
            nodes_in_cluster = [node for node, cluster_id in enumerate(clusters) if cluster_id == i]
            subG = G.subgraph(nodes_in_cluster).copy()
            subgraphs.append(subG)
    
    elif method == 'kernighan-lin':
        num_nodes = len(G.nodes())
        adjacency_list = [list(G.neighbors(i)) for i in range(num_nodes)]
        n_cuts, membership = pymetis.part_graph(num_client, adjacency=adjacency_list)
        subgraphs = []
        for i in range(num_client):
            nodes_in_partition = [node for node, part in enumerate(membership) if part == i]
            subG = G.subgraph(nodes_in_partition).copy()  
            subgraphs.append(subG)
        
    elif method == 'kmeans':
        # 获取图的节点
        nodes = list(G.nodes())
        # 将节点映射为整数索引
        node_index = {node: i for i, node in enumerate(nodes)}
        
        # 获取节点的特征（这里使用节点的度数作为特征）
        X = np.array([list(G.degree(node)) for node in nodes])
        
        # 使用 K-means 聚类
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        labels = kmeans.labels_
        
        # 创建子图列表
        for k in range(K):
            nodes_in_cluster = [nodes[i] for i in range(len(nodes)) if labels[i] == k]
            subgraph = G.subgraph(nodes_in_cluster).copy()
            subgraphs.append(subgraph)
    return subgraphs


def separate_graph_K_clients2(G, num_client, method):
    subgraphs = []
    N = G.number_of_nodes()
    K = num_client
    
    if method == 'evenly_sequence':
            # 将节点映射为整数索引
        nodes = list(G.nodes())    
        # print(nodes)
        # 计算每个簇的节点数
        nodes_per_cluster = N // K        
        # 创建子图列表
        subgraphs = []
        for i in range(K):
            start_index = i * nodes_per_cluster
            end_index = (i + 1) * nodes_per_cluster
            nodes_in_cluster = nodes[start_index:end_index]
            subgraph = G.subgraph(nodes_in_cluster).copy()
            subgraphs.append(subgraph)
        
    elif method == 'spectral_clustering':
        L = nx.normalized_laplacian_matrix(G).toarray()
        n_clusters = num_client
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
        adj_matrix = nx.to_numpy_array(G)
        clusters = spectral_clustering.fit_predict(adj_matrix)
        subgraphs = []
        for i in range(n_clusters):
            nodes_in_cluster = [node for node, cluster_id in enumerate(clusters) if cluster_id == i]
            subG = G.subgraph(nodes_in_cluster).copy()
            subgraphs.append(subG)
        
    elif method == 'kmeans':
        # 获取图的节点
        nodes = list(G.nodes())
        # 将节点映射为整数索引
        node_index = {node: i for i, node in enumerate(nodes)}
        
        # 获取节点的特征（这里使用节点的度数作为特征）
        X = np.array([list(G.degree(node)) for node in nodes])
        
        # 使用 K-means 聚类
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        labels = kmeans.labels_
        
        # 创建子图列表
        for k in range(K):
            nodes_in_cluster = [nodes[i] for i in range(len(nodes)) if labels[i] == k]
            subgraph = G.subgraph(nodes_in_cluster).copy()
            subgraphs.append(subgraph)
            
    return subgraphs