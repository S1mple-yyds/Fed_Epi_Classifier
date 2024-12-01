import torch
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric_temporal.nn.recurrent import TGCN2
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GNN(torch.nn.Module):
    def __init__(self, dim_in, periods, n_classes, batch_size):
        super(GNN, self).__init__()
        embedding_dim = n_classes
        self.n_pred = periods
        self.n_classes = n_classes
        # Embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)

        # self.tgnn = A3TGCN2(in_channels=1, out_channels=32, periods=9, batch_size=batch_size)
        self.tgnn = TGCN2(in_channels=1 * embedding_dim, out_channels=32, batch_size=batch_size)  # 增加了lstm层为128
        self.linear = torch.nn.Linear(32, periods * n_classes)

    def forward(self, data, device):
        x = data.x
        # print(x.shape) #[10000,1]
        x = x.unsqueeze(2)
        # print(x.shape) #[10000,1,1]
        edge_index = data[0].edge_index
        edge_weight = data[0].edge_attr
        prev_hidden_state = None
        # Apply dropout
        # print(edge_index.shape)[2,35500]
        # print(edge_attr.shape)[35500,1]
        # print(x.shape) #[10000.10]
        if device == 'cpu':
            x = torch.FloatTensor(x)
        else:
            x = torch.cuda.FloatTensor(x)
        # Adjust input shape
        batch_size_n_nodes, seq_length, node_feature = x.size()
        batch_size = data.num_graphs
        n_nodes = int(data.num_nodes / batch_size)
        batch_size_n_nodes = batch_size * n_nodes
        node_feature = 1  # Number of features per node

        # Reshape x to fit the LSTM input shape
        x = x.view(batch_size, n_nodes, seq_length, node_feature)
        # print(x.shape) #[50, 200, 10, 1]
        # x = x.squeeze(dim=2)
        # print(x.shape)
        # Transpose dimensions to match LSTM input requirements
        # x = x.unsqueeze(3)
        # print(x.shape) #[50,200,10,1]
        # x = x.permute(0, 1, 3, 2)
        # print(x.shape) #[50,200,1,10]
        x = x.long()

        # Apply embedding layer to the discrete states (0, 1, 2)
        x = self.embedding(x)
        # print(x.shape)
        x = x.reshape(batch_size, n_nodes, seq_length, -1)
        x = torch.squeeze(x[:, :, -1, :])
        # print(x.shape)

        h = self.tgnn(x, edge_index)  # edge_weight, prev_hidden_state) #注意egde_index的数量
        # print(h.shape) #[50,200,lstm_hidden_size]
        # h = torch.squeeze(h[:, -1, :])
        # h = F.relu(h)
        # Apply linear layer
        y = F.relu(h)
        y = self.linear(y)
        # print(y.shape) #[50,200,3]
        # print(h.shape) #[50,200,27]

        s = y.shape
        y = torch.reshape(y, (s[0], n_nodes, self.n_pred, self.n_classes))
        y = torch.reshape(y, (s[0] * n_nodes, self.n_pred, self.n_classes))
        # h = h.view(batch_size_n_nodes, 9, 3)  # (10000, 9, 3)
        # print(y.shape)
        y = F.softmax(y, dim=-1)
        # print(y.shape) #[10000,9,3]
        return y