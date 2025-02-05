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
