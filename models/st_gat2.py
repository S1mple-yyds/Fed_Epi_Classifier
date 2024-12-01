import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ST_GAT2(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """

    def __init__(self, in_channels, n_classes, out_channels, heads=8, dropout=0.0):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param n_classes Number of output classes
        :param n_nodes Number of nodes in the graph
        :param n_pred Number of prediction time points
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super(ST_GAT2, self).__init__()
        #n_classes = 3
        self.n_classes = n_classes
        # n_pred = out_channels
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        #self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.n_hist = in_channels

        lstm1_hidden_size = 32
        lstm2_hidden_size = 64  # origin 128
        embedding_dim = n_classes

        # Embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)

        # Single graph attentional layer with 8 attention heads
        self.gat = GATConv(in_channels=in_channels * embedding_dim, out_channels=in_channels,
                           heads=self.heads, dropout=0.6)

        # self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=embedding_dim*in_channels)

        # Add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.heads, hidden_size=lstm1_hidden_size, batch_first=True)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # self. layer_norm2 = torch.nn.LayerNorm(normalized_shape=lstm1_hidden_size)

        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, batch_first=True)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # Fully-connected neural network
        # self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_classes)
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_pred * self.n_classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data, device):
        """
        Forward pass of the ST-GAT model
        :param data: Data to make a pass on
        :param device: Device to operate on
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = x.long()

        # Apply embedding layer to the discrete states (0, 1, 2)
        x = self.embedding(x)
        # [batch_size * n_nodes, n_hist，embedding_dim]
        # x = F.normalize(x, p=2, dim=-1)

        # Reshape x to [batch_size * n_nodes, embedding_dim * n_hist]
        batch_size = data.num_graphs
        n_node = int(data.num_nodes / batch_size)
        x = torch.reshape(x, (batch_size * n_node, -1))  # [batch_size * n_nodes, embedding_dim * n_hist]

        # Pass through the GATConv layer
        x = self.gat(x, edge_index, edge_attr)
        x = F.elu(x)
        # x.shape: (batch_size * num_nodes, n_hist*head)

        x = x.view(batch_size * n_node, self.n_hist, self.heads)
        # x.shape: (batch_size * num_nodes, n_hist, head)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # x.shape: (batch_size*num_nodes, n_hist,lstm2_hidden_size
        # print(x.shape)
        # Take the output of the last time step
        x = torch.squeeze(x[:, -1, :])
        x = self.linear(x)
        # print(x.shape)
        x = x.view(batch_size * n_node, self.n_pred, self.n_classes)
        # Apply softmax to the class dimension
        x = F.softmax(x, dim=-1)
        # print(x.shape)
        # print(x.shape)
        # [10000,9,3]
        return x
