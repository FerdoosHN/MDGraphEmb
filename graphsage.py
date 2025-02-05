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
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
import numpy as np
import pandas as pd

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

def process_graph(nx_graph):
    for node in nx_graph.nodes:
        nx_graph.nodes[node]['feature'] = [nx_graph.degree[node]]

    data = from_networkx(nx_graph)
    data.x = torch.tensor([nx_graph.nodes[node]['feature'] for node in nx_graph.nodes], dtype=torch.float)
    return data

class GraphSAGEPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.all_embeddings = []
        self.model = None

    def process_graph(self, nx_graph):
        data = process_graph(nx_graph)
        data = data.to(self.device)

        if self.model is None:
            self.model = GraphSAGE(in_channels=data.num_features,
                                   hidden_channels=self.config.HIDDEN_CHANNELS,
                                   out_channels=self.config.OUT_CHANNELS,
                                   num_layers=self.config.NUM_LAYERS).to(self.device)

        self.model.eval()
        with torch.no_grad():
            x = self.model(data.x, data.edge_index)

        embeddings_np = x.cpu().detach().numpy()
        reshaped_embeddings = embeddings_np.reshape(-1)
        self.all_embeddings.append(reshaped_embeddings)

    def save_embeddings(self, config, graphs):
        for nx_graph in graphs:
            self.process_graph(nx_graph)
        df = pd.DataFrame(self.all_embeddings)
        df.to_csv(config.OUTPUT_FILE_GRAPH_SAGE, index=False)
        print(f"GraphSAGE embeddings saved to '{config.OUTPUT_FILE_GRAPH_SAGE}'.")

