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


from config import Config
from protein_data_loader import ProteinDataLoader
from node2vec_embedder import Node2VecEmbedder 
from graphsage import GraphSAGEPipeline
from gcnconv import GCNEmbeddingGenerator
from gatconv import GATEmbeddingGenerator

def run_node2vec(trajectory_file, topology_file, output_file_node2vec):
    config = Config(trajectory_file, topology_file, "", output_file_node2vec, "", "")
    loader = ProteinDataLoader(config)
    graphs = loader.get_graphs()
    node2vec_embedder = Node2VecEmbedder(config)
    node2vec_embedder.save_embeddings(config, graphs)

def run_graphsage(trajectory_file, topology_file, output_file_graph_sage):
    config = Config(trajectory_file, topology_file, output_file_graph_sage, "", "", "")
    loader = ProteinDataLoader(config)
    graphs = loader.get_graphs()
    graphsage_pipeline = GraphSAGEPipeline(config)
    graphsage_pipeline.save_embeddings(config, graphs)

def run_gcn(trajectory_file, topology_file, output_file_gcn):
    config = Config(trajectory_file, topology_file, "", "", output_file_gcn, "")
    loader = ProteinDataLoader(config)
    graphs = loader.get_graphs()
    gcn_generator = GCNEmbeddingGenerator(config)
    gcn_generator.save_embeddings(graphs)

def run_gat(trajectory_file, topology_file, output_file_gat):
    config = Config(trajectory_file, topology_file, "", "", "", output_file_gat)
    loader = ProteinDataLoader(config)
    graphs = loader.get_graphs()
    gat_generator = GATEmbeddingGenerator(config)
    gat_generator.save_embeddings(graphs)

