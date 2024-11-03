from protein_graph import ProteinGraphData

class ProteinDataLoader:
    def __init__(self, config):
        self.config = config
        self.frame_list = None  # Load all frames by default
        self.graph_type = 'distance'  # Specify graph type as needed
        self.protein_graph_data = ProteinGraphData(config.TRAJECTORY_FILE, 
                                                   config.TOPOLOGY_FILE, 
                                                   self.frame_list, 
                                                   self.graph_type)
    
    def get_graphs(self):
        """Returns a list of NetworkX graphs generated from the protein data."""
        graphs = self.protein_graph_data.graph_list
        for i, graph in enumerate(graphs):
            print(f"Graph {i}: Type = {type(graph)}")
        return graphs
