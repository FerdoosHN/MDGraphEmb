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





# config.py

class Config:
    def __init__(self, 
                 trajectory_file, 
                 topology_file, 
                 output_file_graph_sage,
                 output_file_node2vec,
                 output_file_gcn,
                 output_file_gat):
        # Paths
        self.TRAJECTORY_FILE = trajectory_file
        self.TOPOLOGY_FILE = topology_file
        self.OUTPUT_FILE_GRAPH_SAGE = output_file_graph_sage
        self.OUTPUT_FILE_NODE2VEC = output_file_node2vec
        self.OUTPUT_FILE_GCN = output_file_gcn
        self.OUTPUT_FILE_GAT = output_file_gat

        # Additional parameters
        self.NUM_WALKS = 10
        self.WALK_LENGTH = 80
        self.EMBEDDING_SIZE = 64
        self.WINDOW = 10
        self.MIN_COUNT = 1
        self.BATCH_WORDS = 4
        self.DIMENSIONS = 16

        # GraphSAGE parameters
        self.HIDDEN_CHANNELS = 64
        self.OUT_CHANNELS = 64
        self.NUM_LAYERS = 2

        # GCN parameters
        self.GCN_HIDDEN_CHANNELS = 64
        self.GCN_OUT_CHANNELS = 64
        self.GCN_DROPOUT_RATES = [0.5, 0.5, 0.5]
        self.GCN_LEARNING_RATES = [0.01, 0.01, 0.01]
        self.GCN_NUM_EPOCHS = 200
        self.GCN_GRADIENT_CLIP_NORM = 5.0

        # GAT parameters
        self.GAT_HIDDEN_CHANNELS = 64
        self.GAT_OUT_CHANNELS = 64
        self.GAT_DROPOUT_RATES = [0.6, 0.6, 0.6, 0.6]
        self.GAT_LEARNING_RATES = [0.005]
        self.GAT_NUM_EPOCHS = 200
        self.GAT_GRADIENT_CLIP_NORM = 5.0
        self.GAT_NUM_HEADS = [8, 8, 8, 1]
