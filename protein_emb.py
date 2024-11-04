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

