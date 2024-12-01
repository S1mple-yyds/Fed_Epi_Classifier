import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, Data
import shutil
from utils.math_utils import *

"""
We use a real-world traffic speed dataset named PeMSD7. We use the version collected and prepared by Yu et al., 2018 and available here.

The data consists of two files:
PeMSD7_W_228.csv contains the distances between 228 stations across the District 7 of California.
PeMSD7_V_228.csv contains traffic speed collected for those stations in the weekdays of May and June of 2012.
The full description of the dataset can be found in Yu et al., 2018.
"""
def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    """"
    Given distances between all nodes, convert into a weight matrix
    :param W distances
    :param sigma2 User configurable parameter to adjust sparsity of matrix
    :param epsilon User configurable parameter to adjust sparsity of matrix
    :param gat_version If true, use 0/1 weights with self loops. Otherwise, use float
    """
    # n = W.shape[0]
    # W = W / 10000.
    # W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    # # refer to Eq.10
    # W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    #
    # # If using the gat version of this, round to 0/1 and include self loops
    # if gat_version:
    #     W[W>0] = 1
    #     W += np.identity(n)

    return W

class TrafficDataset(InMemoryDataset):
    """
    Dataset for Graph Neural Networks for traffic congestion classification.
    """
    def __init__(self, config, W, client_id, root='', transform=None, pre_transform=None):
        self.config = config
        self.W = W
        self.client_id = client_id
        super().__init__(root, transform, pre_transform)

        # 删除之前保存的数据文件
        if os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])

        # 删除原来的 raw 文件
        '''raw_files = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir)]
        for file in raw_files:
            if os.path.exists(file):
                os.remove(file)'''
        # 调用process生成新数据
        self.process()
        self.data, self.slices, self.n_node = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'client{self.client_id}_node_state_tiny.csv']

    @property
    def processed_file_names(self):
        return [f'data_{self.client_id}.pt']

    def download(self):
        #copyfile('./dataset/node_state_tiny.csv', os.path.join(self.raw_dir, 'node_state_tiny.csv'))
        # Copy the specific node state file into the raw directory
        src_path = os.path.join('./ClientDatas/', f'client{self.client_id}_node_state_tiny.csv')
        dst_path = os.path.join(self.raw_dir, f'client{self.client_id}_node_state_tiny.csv')
        shutil.copyfile(src_path, dst_path)

    def get_n_node(self):
    
        return self.n_node

    def process(self):
        """
        Process the raw datasets into saved .pt dataset for later use.
        Note that any self.fields here wont exist if loading straight from the .pt file
        """
        # Data Preprocessing and loading
        #data = pd.read_csv(self.raw_file_names[0], header=None).values
        # Data Preprocessing and loading
        self.download()
        filename = f'client{self.client_id}_node_state_tiny.csv'
        data = pd.read_csv(os.path.join(self.raw_dir, filename), header=None).values

        _, n_node = data.shape
        n_window = self.config['N_PRED'] + self.config['N_HIST']

        # manipulate nxn matrix into 2xnum_edges
        edge_index = torch.zeros((2, n_node**2), dtype=torch.long)
        edge_attr = torch.zeros((n_node**2, 1))
        num_edges = 0
        for i in range(n_node):
            for j in range(n_node):
                if self.W[i, j] != 0.:
                    edge_index[0, num_edges] = i
                    edge_index[1, num_edges] = j
                    edge_attr[num_edges] = self.W[i, j]
                    num_edges += 1
        edge_index = edge_index[:, :num_edges]
        edge_attr = edge_attr[:num_edges]

        sequences = []

        # T x F x N
        # for i in range(self.config['N_DAYS']):
        #     for j in range(self.config['N_SLOT']):
        #         # for each time point construct a different graph with data object
        #         g = Data()
        #         g.__num_nodes__ = n_node
        
        #         g.edge_index = edge_index
        #         g.edge_attr = edge_attr
        
        #         # (F,N) switched to (N,F)
        #         sta = i * self.config['N_DAY_SLOT'] + j
        #         end = sta + n_window
        #         full_window = np.swapaxes(data[sta:end, :], 0, 1)
        #         g.x = torch.FloatTensor(full_window[:, 0:self.config['N_HIST']].astype(float))
        #         g.y = torch.FloatTensor(full_window[:, self.config['N_HIST']::].astype(float))
        #         sequences += [g]

        total_time_slots = self.config['N_DAYS'] * self.config['N_DAY_SLOT']
        for t in range(total_time_slots - n_window + 1):
            # 构建一个图形数据对象
            g = Data()
            g.__num_nodes__ = n_node

            g.edge_index = edge_index
            g.edge_attr = edge_attr
            # 定义滑动窗口的起始和结束位置
            full_window = np.swapaxes(data[t:t + n_window, :], 0, 1)

            # 将前N_HIST个时间点作为输入x，后面的时间点作为标签y
            g.x = torch.FloatTensor(full_window[:, 0:self.config['N_HIST']].astype(float))
            g.y = torch.FloatTensor(full_window[:, self.config['N_HIST']::].astype(float))
            sequences += [g]

        # Make the actual dataset
        data, slices = self.collate(sequences)
        torch.save((data, slices, n_node), self.processed_paths[0])

def get_splits(dataset: TrafficDataset, n_slot, splits):
    """
    Given the data, split it into random subsets of train, val, and test as given by splits
    :param dataset: TrafficDataset object to split
    :param n_slot: Number of possible sliding windows in a day
    :param splits: (train, val, test) ratios
    """
    split_train, split_val, split_test = splits
    #total_split = split_train + split_val + split_test
    #assert abs(total_split - 1.0) < 1e-6, "The sum of the splits must equal 1."
    i = n_slot*split_train
    j = n_slot*split_val
    train = dataset[:i]
    val = dataset[i:i+j]
    test = dataset[i+j:]
    whole = dataset[:]

    return train, val, test, whole

def get_splits_by_ratio(dataset: TrafficDataset, splits):
    total_len = len(dataset)
    split_train, split_val, split_test = splits

    train_size = int(split_train * total_len)
    val_size = int(split_val * total_len)
    test_size = total_len - train_size - val_size

    # 按顺序切分数据集
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:]
    whole = dataset[:]

    return train_set, val_set, test_set, whole

def get_splits_by_ratio2(dataset: TrafficDataset, splits):
    total_len = len(dataset)
    split_train, split_val, split_test_ratio = splits

    train_size = int(split_train * total_len)
    val_size = int(split_val * total_len)
    test_size = int(split_test_ratio * total_len)

    # 基于训练集和验证集后的剩余部分计算测试集的大小
    #remaining_size = total_len - train_size - val_size
    #test_size = int(remaining_size * split_test_ratio)

    # 由于我们是从剩余部分计算测试集大小，所以这里需要确保测试集大小不超过剩余部分
    #if test_size > remaining_size:
    #    test_size = remaining_size

    # 更新验证集大小以确保总和正确
    #val_size = remaining_size - test_size

    print(train_size, val_size, test_size)
    # 按顺序切分数据集
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:train_size + val_size + test_size]
    whole = dataset[:]

    return train_set, val_set, test_set, whole

