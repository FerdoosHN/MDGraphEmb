import networkx as nx
import numpy as np
from protein import Protein

class ProteinGraph:
    """
    This class is responsible for generating graph representations of protein data.
    """
    def __init__(self, protein):
        """
        Initialize the ProteinGraph object.

        Parameters
        ----------
        protein : Protein
            Protein object containing trajectory and topology data.
        """
        self.protein = protein
    
    def get_distance_contact_matrix_graph(self, cutoff=10, frame=0, adj_mat=None):
        """ 
        Generate graphs corresponding to the adjacency matrix based on 
        the given cut-off 10 Å on distance with contact approach.
        
        Parameters
        ----------
        cutoff : float, optional
            Distance cut-off 10 Å for edges in the graph (default is 10).
        frame : int, optional
            Frame index to use for the graph generation (default is 0).
        adj_mat : numpy.ndarray, optional
            Adjacency matrix to use (default is None, which means it will be computed).

        Returns
        -------
        networkx.Graph
            Graph representation of the distance matrix with edges below the cut-off.
        """
        if adj_mat is None:
            adj_mat = self.protein.get_c_alpha_distance_matrix(frame)
        adj_mat[adj_mat > cutoff] = 0
        c_alpha_distance_contact_graph = nx.from_numpy_array(np.array(adj_mat))
        return c_alpha_distance_contact_graph

    def get_distance_matrix_graph(self, cutoff=10, frame=0, adj_mat=None):
        """ 
        Generate graphs corresponding to the adjacency matrix based on 
        the given cut-off 10 Å on distance considering the weighted edges.
        
        Parameters
        ----------
        cutoff : float, optional
            Distance cut-off 10 Å for edges in the graph (default is 10).
        frame : int, optional
            Frame index to use for the graph generation (default is 0).
        adj_mat : numpy.ndarray, optional
            Adjacency matrix to use (default is None, which means it will be computed).

        Returns
        -------
        networkx.Graph
            Graph representation of the distance matrix with weighted edges below the cut-off.
        """
        if adj_mat is None:
            adj_mat = self.protein.get_c_alpha_distance_matrix(frame)
        
        # Apply the cutoff and transform distances
        transformed_adj_mat = np.where(adj_mat > cutoff, 0, 1 - (adj_mat / cutoff))
        np.fill_diagonal(transformed_adj_mat, 0)
        
        # Create the graph from the transformed adjacency matrix
        c_alpha_distance_graph = nx.from_numpy_array(transformed_adj_mat)
        
        return c_alpha_distance_graph

class ProteinGraphData:
    def __init__(self, trajectory_filename, topology_filename, frame_list=None, graph_type='both'):
        """
        Initialize the ProteinGraphData object.

        Parameters
        ----------
        trajectory_filename : str
            Path to the trajectory file.
        topology_filename : str
            Path to the topology file.
        frame_list : list of int, optional
            List of frame indices to process (default is None, which processes all frames).
        graph_type : str, optional
            Type of graph to generate: 'contact', 'distance', or 'both' (default is 'both').
        """
        self.trajectory_filename = trajectory_filename
        self.topology_filename = topology_filename
        self.frame_list = frame_list
        self.graph_list = []
        self.graph_type = graph_type.lower()
        self.protein = Protein(self.trajectory_filename, self.topology_filename)
        self.protein_graph = ProteinGraph(self.protein)
        self.n_frame = self.protein.n_frame

        if self.frame_list is None:
            self.frame_list = range(0, self.n_frame)
        
        self.collect_graphs()
        self.n_graph = len(self.graph_list)

    def collect_graphs(self):
        """
        Collects graphs for the specified frames based on the selected graph type.
        """
        for frame_id in self.frame_list:
            if self.graph_type in ['distance', 'both']:
                distance_matrix_graph = self.protein_graph.get_distance_matrix_graph(frame=frame_id)
                self.graph_list.append(distance_matrix_graph)
            
            if self.graph_type in ['contact', 'both']:
                contact_matrix_graph = self.protein_graph.get_distance_contact_matrix_graph(frame=frame_id)
                self.graph_list.append(contact_matrix_graph)
        
        return self.graph_list

