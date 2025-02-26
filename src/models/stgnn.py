import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class STGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, num_timesteps):
        super(STGNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.gcn1 = GCNConv(in_channels, 64)
        self.gcn2 = GCNConv(64, 128)
        self.gcn3 = GCNConv(128, out_channels)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, out_channels)
    
    def forward(self, x, edge_index):
        # x: node features, shape: [num_timesteps, num_nodes, in_channels]
        batch_size = x.size(0)

        x = x.view(-1, x.size(-1))  # Flattening the time dimension
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)

        # Reshape back for LSTM input
        x = x.view(batch_size, self.num_timesteps, -1)  # Shape: [batch_size, num_timesteps, 128]
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Only use the last timestep for prediction
        
        return x
