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
