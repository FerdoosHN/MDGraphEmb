"""
@release_date  : $release_date
@version       : $release_version
@author        : Ferdoos Hossein Nezhad

This file is part of the MDGraphEMB software 
(https://github.com/FerdoosHN/MDGraphEMB).
Copyright (c) 2025 Ferdoos Hossein Nezhad and Alessandro Pandini.

This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rates):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.dropout_rates = dropout_rates

    def forward(self, x, edge_index):
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=self.dropout_rates[0], training=self.training)

        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)) + x1)
        x2 = F.dropout(x2, p=self.dropout_rates[1], training=self.training)

        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)) + x2)
        x3 = F.dropout(x3, p=self.dropout_rates[2], training=self.training)

        x4 = self.conv4(x3, edge_index)
        return x4

class GCNEmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_pyg_data(self, graph):
        edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        x = torch.tensor(np.array([graph.nodes[node].get('feature', [1]) for node in graph.nodes()]), dtype=torch.float)
        return Data(x=x, edge_index=edge_index).to(self.device)

    def generate_gcn_node_embeddings(self, graph):
        data = self.to_pyg_data(graph)
        model = GCN(data.num_node_features, self.config.GCN_HIDDEN_CHANNELS, self.config.GCN_OUT_CHANNELS, 
                    self.config.GCN_DROPOUT_RATES).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LEARNING_RATES[0])
        criterion = nn.CrossEntropyLoss()

        labels = torch.tensor([graph.nodes[node].get('label', 0) for node in graph.nodes()], dtype=torch.long).to(self.device)
        
        model.train()
        for epoch in range(self.config.GCN_NUM_EPOCHS):
            optimizer.zero_grad()
            node_embeddings = model(data.x, data.edge_index)
            loss = criterion(node_embeddings, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.GCN_GRADIENT_CLIP_NORM)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            node_embeddings = model(data.x, data.edge_index)
            embeddings = node_embeddings.cpu().numpy()

        return embeddings

    def save_embeddings(self, graphs):
        all_embeddings = []
        for graph in graphs:
            embeddings = self.generate_gcn_node_embeddings(graph)
            flattened_embeddings = embeddings.reshape(-1)
            all_embeddings.append(flattened_embeddings)
        
        df = pd.DataFrame(all_embeddings)
        df.to_csv(self.config.OUTPUT_FILE_GCN, index=False)
        print(f"GCN embeddings saved to '{self.config.OUTPUT_FILE_GCN}'.")

