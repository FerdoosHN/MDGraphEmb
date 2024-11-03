import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import self_distance_array

class Protein:
    """ 
    Overview:
    This class is used to represent protein data.
    ------------
    Attributes:
    trajectory_filename: str (Path to trajectory file.)
    topology_filename: str (Path to topology file.)
    """
 
    def __init__(self, trajectory_filename, topology_filename):
        self.trajectory_filename = trajectory_filename
        self.topology_filename = topology_filename
        self.trajectory_data = self.get_trajectory(topology_filename, trajectory_filename)
        self.c_alpha = self.get_c_alpha()
        self.n_frame = len(self.trajectory_data.trajectory)
        self.n_atoms = self.trajectory_data.atoms.n_atoms
        self.n_c_alpha = len(self.c_alpha)

    def get_trajectory(self, topology_filename, trajectory_filename):
        """
        Get trajectory and topology files into Universe to build the object.

        Parameters
        ----------
        trajectory_filename : str
            Path to trajectory file.
        topology_filename : str
            Path to topology file.

        Returns
        -------
        mda.Universe
            Universe object containing trajectory and topology data.
        """
        trajectory_data = mda.Universe(topology_filename, trajectory_filename)
        return trajectory_data
    
    def get_c_alpha(self, frame=0):
        """ 
        Select Calpha atoms based on the specific frame in the protein trajectory.
        
        Parameters
        ----------
        frame : int, optional
            Frame index to select (default is 0).

        Returns
        -------
        mda.core.groups.AtomGroup
            AtomGroup of Calpha atoms.
        """
        self.trajectory_data.trajectory[frame]
        c_alpha = self.trajectory_data.select_atoms("name CA")
        return c_alpha   
    
    def get_c_alpha_distance_matrix(self, frame=0):
        """ 
        Create the adjacency matrix for the Calpha atoms based on the 
        self_distance_array function from the MDAnalysis library.
        
        Parameters
        ----------
        frame : int, optional
            Frame index to use for the distance calculation (default is 0).

        Returns
        -------
        numpy.ndarray
            Adjacency matrix of distances between Calpha atoms.
        """
        c_alpha = self.get_c_alpha(frame) 
        c_alpha_distance_matrix = np.zeros((self.n_c_alpha, self.n_c_alpha))
        triu = np.triu_indices_from(c_alpha_distance_matrix, k=1)
        distance_array = self_distance_array(c_alpha.positions)
        c_alpha_distance_matrix[triu] = distance_array
        c_alpha_distance_matrix.T[triu] = distance_array 
        return c_alpha_distance_matrix

