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



import numpy as np
import pandas as pd
from node2vec import Node2Vec

class Node2VecEmbedder:
    def __init__(self, config):
        self.num_walks = config.NUM_WALKS
        self.walk_length = config.WALK_LENGTH
        self.embedding_size = config.EMBEDDING_SIZE
        self.window = config.WINDOW
        self.min_count = config.MIN_COUNT
        self.batch_words = config.BATCH_WORDS
        self.dimensions = config.DIMENSIONS

    def generate_embedding_for_graph(self, graph):
        node2vec = Node2Vec(graph, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks)
        model = node2vec.fit(window=self.window, min_count=self.min_count, batch_words=self.batch_words)
        embeddings = np.array([model.wv[str(node)] for node in graph.nodes()])
        return embeddings

    def save_embeddings(self, config, graphs):
        all_embeddings = []
        for graph in graphs:
            embeddings = self.generate_embedding_for_graph(graph)
            reshaped_embeddings = embeddings.reshape(-1)  # Modified to reshape to a 1D array
            all_embeddings.append(reshaped_embeddings)
        df = pd.DataFrame(all_embeddings)
        df.to_csv(config.OUTPUT_FILE_NODE2VEC, index=False)
        print(f"Node embeddings saved to '{config.OUTPUT_FILE_NODE2VEC}'.")











