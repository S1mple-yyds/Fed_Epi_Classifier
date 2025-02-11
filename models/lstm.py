import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class LSTM(torch.nn.Module):


    def __init__(self, in_channels, n_classes, out_channels, heads=8, dropout=0.0):

        super(LSTM, self).__init__()
        #n_classes = 2
        self.n_classes = n_classes

        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        #self.n_nodes = n_nodes

        lstm1_hidden_size = 16
        lstm2_hidden_size = 64

        embedding_dim = n_classes

        # Embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)
        

        # # Single graph attentional layer with 8 attention heads
        # self.gat = GATConv(in_channels=embedding_dim*in_channels, out_channels=embedding_dim*in_channels,
        #                    heads=heads, dropout=0, concat=False)

        # self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=embedding_dim*in_channels)

        # Add two LSTM layers
        #self.lstm1 = torch.nn.LSTM(input_size=embedding_dim, hidden_size=lstm1_hidden_size, batch_first=True)
        self.lstm1 = torch.nn.LSTM(input_size=in_channels, hidden_size=lstm1_hidden_size, batch_first=True)
        # self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        #self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, batch_first=True)
        #for name, param in self.lstm2.named_parameters():
        #    if 'bias' in name:
        #        torch.nn.init.constant_(param, 0.0)
       #     elif 'weight' in name:
       #         torch.nn.init.xavier_uniform_(param)

        # Fully-connected neural network
        # self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_classes)
        #self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_pred * self.n_classes)
        self.linear = torch.nn.Linear(lstm1_hidden_size, self.n_pred * self.n_classes)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data, device):

        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        n_node = int(data.num_nodes / batch_size)

        #x = x.long()

        # Apply embedding layer to the discrete states (0, 1, 2)
        #x = self.embedding(x)
        # [batch*nodes,n_hist, embed_dim]
        # x = F.normalize(x, p=2, dim=-1)

        # x = self.layer_norm2(x)
        x, _ = self.lstm1(x)
        #x, _ = self.lstm2(x)

        #x = torch.squeeze(x[:, -1, :])
        x = self.linear(x)
        # print(x.shape)
        x = x.view(batch_size * n_node, self.n_pred, self.n_classes)
        # Apply softmax to the class dimension
        x = F.softmax(x, dim=-1)
        # print(x.shape)

        return x
