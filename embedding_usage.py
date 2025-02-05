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



import sys
import os

# Determine the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)  # Add project root to sys.path for imports

import protein_emb

# Instructions for the user
print("Generate GraphSAGE embeddings by providing the required file paths.\n")

# Define the required paths relative to the project root
trajectory_file = os.path.join(project_root, 'ADKData', 'R02_4AKE_DM_prod_T300_protein_dt200.xtc')
topology_file = os.path.join(project_root, 'ADKData', 'R02_4AKE_DM_prod_T300_protein.gro')
output_file_graph_sage = os.path.join(project_root, 'embedded_data','graphsage_embeddings.csv')

# Run the GraphSAGE embedding generation
protein_emb.run_graphsage(trajectory_file, topology_file, output_file_graph_sage)

print("\nGraphSAGE embedding generation completed successfully.")

